# Free Interview (DeepSeek R1)

Self-hosted, free mock interview app using an OpenAI-compatible DeepSeek R1 API. No paywall. Upload your CV, paste JD, generate questions, get ratings & feedback, and stream suggested answers. Includes browser TTS and optional STT.

## Quick start

1. Copy environment example

```
cp .env.example .env.local
# Optionally set OPENROUTER_API_KEY or DEEPSEEK_API_KEY
```

2. Install deps and run dev

```
npm install
npm run dev
```

3. Open http://localhost:3000 and go to Settings to paste your API key if you did not set it server-side.

## Features

- CV/JD parsing: PDF, DOCX, TXT
- Question generation grounded in CV/JD
- Answer evaluation: score, strengths, improvements, summary
- Suggested answers: streaming
- Real-time mock: TTS (speechSynthesis) + STT (webkitSpeechRecognition when supported)
- No paywall, no time limits

## Configuration

- LLM_BASE_URL: default `https://openrouter.ai/api/v1` (works with DeepSeek R1 via OpenRouter)
- LLM_MODEL: default `deepseek/deepseek-r1:free`
- Alternative DeepSeek official: set `LLM_BASE_URL=https://api.deepseek.com/v1` and `LLM_MODEL=deepseek-reasoner`

You can provide API keys either server-side via `.env.local` or client-side in Settings (stored in `localStorage`). Requests forward the `x-api-key` header.

## Notes

- STT support varies by browser. Chrome-based browsers usually provide `webkitSpeechRecognition`.
- This is a reference clone-style app; design is intentionally minimal.
