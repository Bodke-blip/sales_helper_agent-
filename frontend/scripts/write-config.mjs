import { writeFileSync } from "node:fs";

const apiBaseUrl = (
  process.env.PREDIKLY_API_BASE_URL ||
  process.env.VITE_API_BASE_URL ||
  ""
).replace(/\/$/, "");

writeFileSync(
  new URL("../config.js", import.meta.url),
  `window.PREDIKLY_API_BASE_URL = ${JSON.stringify(apiBaseUrl)};\n`,
);
