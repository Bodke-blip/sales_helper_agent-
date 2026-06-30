import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from agents.prompt_management import ManagedPrompt
from agents.list_of_agent import (
    ALL_PREDIKLY_USECASES_PATH,
    ListOfAgent,
    load_all_predikly_usecases,
)
from agents.main_orchestrator_agent import (
    MAIN_ORCHESTRATOR_PROMPT,
    MainOrchestratorAgent,
)
from agents.sales_helper_agent import (
    AGENT_RECURSION_LIMIT,
    SALES_HELPER_SYSTEM_PROMPT,
    build_specialist_tools,
    invoke_specialist_agent,
    sales_helper_agent_node,
)
from agents.specialist_agents import (
    SPECIALIST_AGENT_CLASSES,
    SPECIALIST_SPECS,
    SpecialistAgent,
    create_specialist_agent,
)
from agents.tracing import agent_observation, trace_event


class MultiAgentArchitectureTests(unittest.TestCase):
    @patch("agents.tracing.langfuse_client")
    @patch("agents.tracing.langfuse_enabled", return_value=True)
    def test_langfuse_agent_observation_uses_agent_type(
        self,
        _langfuse_enabled,
        langfuse_client,
    ):
        observation = Mock()
        context = MagicMock()
        context.__enter__.return_value = observation
        langfuse_client.start_as_current_observation.return_value = context

        with agent_observation(
            {"trace_id": "trace-test"},
            name="BenefitsAgent",
            input_data={"query": "benefits"},
            metadata={"node_type": "specialist_agent"},
            model="test-model",
        ) as active_observation:
            self.assertIs(active_observation, observation)

        langfuse_client.start_as_current_observation.assert_called_once()
        call_kwargs = langfuse_client.start_as_current_observation.call_args.kwargs
        self.assertEqual(call_kwargs["name"], "BenefitsAgent")
        self.assertEqual(call_kwargs["as_type"], "agent")

    @patch("agents.tracing.langfuse_client")
    @patch("agents.tracing.langfuse_enabled", return_value=True)
    def test_langfuse_retrieval_event_uses_retriever_type(
        self,
        _langfuse_enabled,
        langfuse_client,
    ):
        langfuse_client.start_observation.return_value = Mock()

        trace_event(
            {"trace_id": "trace-test"},
            name="tool.hybrid_retrieval.complete",
            input_data={"query": "case study"},
            output_data={"sources_count": 2},
            metadata={"observation_type": "retriever"},
        )

        call_kwargs = langfuse_client.start_observation.call_args.kwargs
        self.assertEqual(call_kwargs["as_type"], "retriever")

    def test_expected_four_specialists_are_registered(self):
        self.assertEqual(
            set(SPECIALIST_SPECS),
            {
                "list_of_agent",
                "benefits_agent",
                "usecase_agent",
                "customer_domain_agent",
            },
        )
        self.assertEqual(
            len({agent_class.__module__ for agent_class in SPECIALIST_AGENT_CLASSES.values()}),
            4,
        )

    def test_complete_catalog_trigger_is_narrow_and_snapshot_is_fixed(self):
        self.assertTrue(ALL_PREDIKLY_USECASES_PATH.is_file())
        self.assertEqual(len(load_all_predikly_usecases()), 60)
        self.assertTrue(
            ListOfAgent.matches_complete_catalog_request(
                "List all use cases ever uploaded about Predikly"
            )
        )
        self.assertTrue(ListOfAgent.matches_complete_catalog_request("List all use cases"))
        self.assertFalse(
            ListOfAgent.matches_complete_catalog_request("List all healthcare use cases")
        )
        self.assertFalse(
            ListOfAgent.matches_complete_catalog_request("List all use cases for Infinx")
        )
        answer = ListOfAgent.complete_catalog_answer()
        self.assertIn("contains 60 entries", answer)
        self.assertIn("60. TRA (Tidal Reporting App) — Tidal", answer)

    @patch("agents.sales_helper_agent.trace_event")
    @patch("agents.sales_helper_agent.create_specialist_agent")
    def test_complete_catalog_request_bypasses_llm_and_qdrant(
        self,
        create_specialist,
        _trace_event,
    ):
        state, _response, agent_run = invoke_specialist_agent(
            base_state={
                "user_query": "List all use cases ever uploaded about Predikly",
                "internal_context": [],
                "qdrant_sources": [],
            },
            spec=SPECIALIST_SPECS["list_of_agent"],
            model=object(),
            model_name="test-model",
            request="List all use cases ever uploaded about Predikly",
        )

        create_specialist.assert_not_called()
        self.assertEqual(agent_run["status"], "hardcoded_catalog")
        self.assertEqual(state["retrieval_cache_status"], "hardcoded_catalog")
        self.assertEqual(
            state["hardcoded_all_usecases_answer"],
            ListOfAgent.complete_catalog_answer(),
        )

    def test_every_specialist_has_shared_retrieval_and_profile_tools(self):
        expected_profile_tool = {
            "list_of_agent": "list_or_count_internal_usecases",
            "benefits_agent": "search_benefits_evidence",
            "usecase_agent": "search_usecase_details",
            "customer_domain_agent": "search_customer_domain_knowledge",
        }

        for key, spec in SPECIALIST_SPECS.items():
            tools = build_specialist_tools({}, spec, {})
            tool_names = {item.name for item in tools}
            self.assertIn("search_internal_knowledge", tool_names)
            self.assertIn(expected_profile_tool[key], tool_names)

    @patch("agents.specialist_base.load_managed_prompt")
    def test_every_specialist_class_explicitly_uses_create_agent(self, load_prompt):
        model = object()
        tools = [Mock()]

        for key, agent_class in SPECIALIST_AGENT_CLASSES.items():
            with self.subTest(agent=key):
                spec = SPECIALIST_SPECS[key]
                managed = ManagedPrompt(spec.fallback_prompt, spec.prompt_name, 3, "langfuse")
                load_prompt.return_value = managed

                with patch(f"{agent_class.__module__}.create_agent") as create_agent_mock:
                    create_agent_mock.return_value = object()
                    specialist = agent_class(model=model, tools=tools)
                    self.assertIsInstance(specialist, SpecialistAgent)
                    _, returned_prompt = specialist.create()

                    self.assertEqual(returned_prompt, managed)
                    create_agent_mock.assert_called_once_with(
                        model=model,
                        tools=tools,
                        system_prompt=managed.text,
                        name=key,
                    )

    @patch("agents.main_orchestrator_agent.create_agent")
    @patch("agents.main_orchestrator_agent.load_managed_prompt")
    def test_main_orchestrator_builder_explicitly_uses_create_agent(
        self,
        load_prompt,
        create_agent_mock,
    ):
        managed = ManagedPrompt(
            SALES_HELPER_SYSTEM_PROMPT,
            "predikly/agents/main-orchestrator",
            2,
            "langfuse",
        )
        load_prompt.return_value = managed
        model = object()
        tools = [Mock()]

        _, returned_prompt = MainOrchestratorAgent(model=model, tools=tools).create()

        self.assertEqual(returned_prompt, managed)
        create_agent_mock.assert_called_once_with(
            model=model,
            tools=tools,
            system_prompt=managed.text,
            name="main_orchestrator_agent",
        )

    @patch("agents.specialist_agents.SPECIALIST_AGENT_CLASSES")
    def test_specialist_dispatch_uses_registered_agent_class(self, agent_classes):
        spec = SPECIALIST_SPECS["benefits_agent"]
        expected = (object(), ManagedPrompt("prompt", spec.prompt_name, None, "local_fallback"))
        agent_instance = Mock()
        agent_instance.create.return_value = expected
        agent_class = Mock(return_value=agent_instance)
        agent_classes.get.return_value = agent_class
        model = object()
        tools = [Mock()]

        result = create_specialist_agent(model=model, spec=spec, tools=tools)

        self.assertEqual(result, expected)
        agent_class.assert_called_once_with(model=model, tools=tools)
        agent_instance.create.assert_called_once_with()

    def test_orchestrator_prompt_has_explicit_specialist_boundaries(self):
        self.assertIn("ask_list_of_agent", SALES_HELPER_SYSTEM_PROMPT)
        self.assertIn("ask_benefits_agent", SALES_HELPER_SYSTEM_PROMPT)
        self.assertIn("ask_usecase_agent", SALES_HELPER_SYSTEM_PROMPT)
        self.assertIn("ask_customer_domain_agent", SALES_HELPER_SYSTEM_PROMPT)
        self.assertIn("Do not call BenefitsAgent unless", SALES_HELPER_SYSTEM_PROMPT)

    @patch("agents.sales_helper_agent.invoke_specialist_agent")
    @patch("agents.sales_helper_agent.agent_observation", return_value=nullcontext(None))
    @patch("agents.main_orchestrator_agent.create_agent")
    @patch("agents.main_orchestrator_agent.load_managed_prompt")
    @patch("agents.sales_helper_agent.get_secondary_llm", return_value=None)
    @patch("agents.sales_helper_agent.get_primary_llm")
    @patch("agents.sales_helper_agent.gemini_enabled", return_value=True)
    def test_supervisor_routes_to_specialist_and_propagates_state(
        self,
        _gemini_enabled,
        get_primary_llm,
        _get_secondary_llm,
        load_prompt,
        create_agent_mock,
        _agent_observation,
        invoke_specialist,
    ):
        model = object()
        get_primary_llm.return_value = model
        load_prompt.return_value = ManagedPrompt(
            SALES_HELPER_SYSTEM_PROMPT,
            "predikly/agents/main-orchestrator",
            4,
            "langfuse",
        )
        agent_run = {
            "agent": "benefits_agent",
            "display_name": "BenefitsAgent",
            "status": "grounded",
            "sources_count": 1,
        }

        def fake_specialist(**kwargs):
            specialist_state = {
                **kwargs["base_state"],
                "internal_context": [{"text": "Reduced manual processing."}],
                "qdrant_sources": [{"ppt_name": "Case Study.pptx", "score": 0.9}],
                "selected_agents": ["benefits_agent", "knowledge_retrieval", "eval"],
                "agent_runs": [agent_run],
            }
            return specialist_state, '{"answer":"Reduced manual processing."}', agent_run

        invoke_specialist.side_effect = fake_specialist

        class FakeSupervisor:
            def __init__(self, tools):
                self.tools = {item.name: item for item in tools}

            def invoke(self, _input, config=None):
                self.config = config
                self.tools["ask_benefits_agent"].invoke(
                    {"request": "Return the documented benefits."}
                )
                return {"messages": [SimpleNamespace(content="Reduced manual processing.")]}

        created_supervisors = []

        def build_fake_supervisor(**kwargs):
            supervisor = FakeSupervisor(kwargs["tools"])
            created_supervisors.append(supervisor)
            return supervisor

        create_agent_mock.side_effect = build_fake_supervisor
        result = sales_helper_agent_node(
            {
                "user_query": "What are the benefits?",
                "contextual_query": "What are the benefits?",
                "chat_history": [],
                "internal_context": [],
                "qdrant_sources": [],
            }
        )

        self.assertEqual(result["orchestrator_tool"], "benefits_agent")
        self.assertIn("benefits_agent", result["selected_agents"])
        self.assertEqual(result["agent_runs"][0]["display_name"], "BenefitsAgent")
        self.assertEqual(result["final_response"]["answer"], "Reduced manual processing.")
        self.assertEqual(created_supervisors[0].config, {"recursion_limit": AGENT_RECURSION_LIMIT})


if __name__ == "__main__":
    unittest.main()
