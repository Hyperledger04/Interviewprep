"use client";

import { useEffect, useMemo, useRef, useState } from 'react';

type Question = {
  id: string;
  question: string;
  category: string;
  difficulty: 'easy' | 'medium' | 'hard';
};

type Evaluation = {
  score: number;
  strengths: string[];
  improvements: string[];
  summary: string;
  suggested_answer: string;
};

// Local minimal Web Speech types to avoid relying on global ambient types
interface LocalSpeechRecognitionAlternative {
  transcript: string;
  confidence?: number;
}
interface LocalSpeechRecognitionResult {
  isFinal: boolean;
  readonly length: number;
  [index: number]: LocalSpeechRecognitionAlternative;
}
interface LocalSpeechRecognitionResultList {
  readonly length: number;
  item(index: number): LocalSpeechRecognitionResult;
  [index: number]: LocalSpeechRecognitionResult;
}
interface LocalSpeechRecognitionEvent extends Event {
  readonly resultIndex: number;
  readonly results: LocalSpeechRecognitionResultList;
}
interface LocalSpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  onend: ((this: LocalSpeechRecognition, ev: Event) => void) | null;
  onstart: ((this: LocalSpeechRecognition, ev: Event) => void) | null;
  onresult: ((this: LocalSpeechRecognition, ev: LocalSpeechRecognitionEvent) => void) | null;
}

type SpeechRecognitionConstructor = new () => LocalSpeechRecognition;

declare global {
  interface Window {
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
    SpeechRecognition?: SpeechRecognitionConstructor;
  }
}

function useSpeechSynthesis() {
  const synthRef = useRef<SpeechSynthesis | null>(null);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    synthRef.current = window.speechSynthesis;
    const update = () => setVoices(window.speechSynthesis.getVoices());
    update();
    window.speechSynthesis.onvoiceschanged = update;
  }, []);

  const speak = (text: string, voice?: SpeechSynthesisVoice) => {
    if (!synthRef.current) return;
    const utter = new SpeechSynthesisUtterance(text);
    if (voice) utter.voice = voice;
    utter.rate = 1.0;
    utter.pitch = 1.0;
    synthRef.current.cancel();
    synthRef.current.speak(utter);
  };

  const cancel = () => synthRef.current?.cancel();

  return { voices, speak, cancel };
}

function useSpeechRecognition() {
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const recognitionRef = useRef<LocalSpeechRecognition | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const SR = window.webkitSpeechRecognition || window.SpeechRecognition;
    if (!SR) return;
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = 'en-US';
    rec.onstart = () => setListening(true);
    rec.onend = () => setListening(false);
    rec.onresult = (event: LocalSpeechRecognitionEvent) => {
      let interim = '';
      let final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        if (res.isFinal) final += res[0].transcript;
        else interim += res[0].transcript;
      }
      setTranscript(final || interim);
    };
    recognitionRef.current = rec;
    setSupported(true);
  }, []);

  const start = () => recognitionRef.current?.start?.();
  const stop = () => recognitionRef.current?.stop?.();
  const reset = () => setTranscript('');
  return { supported, listening, transcript, start, stop, reset };
}

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<'upload' | 'generate' | 'mock' | 'settings'>('upload');
  const [cvText, setCvText] = useState('');
  const [jdText, setJdText] = useState('');
  const [questions, setQuestions] = useState<Question[]>([]);
  const [numQuestions, setNumQuestions] = useState(10);

  const [currentQuestionIdx, setCurrentQuestionIdx] = useState<number>(0);
  const [candidateAnswer, setCandidateAnswer] = useState('');
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [suggestedAnswer, setSuggestedAnswer] = useState('');

  const { speak } = useSpeechSynthesis();
  const { supported: sttSupported, listening, transcript, start, stop, reset } = useSpeechRecognition();

  useEffect(() => {
    if (transcript) setCandidateAnswer(transcript);
  }, [transcript]);

  const currentQuestion = useMemo(() => questions[currentQuestionIdx], [questions, currentQuestionIdx]);

  async function handleExtract(files: FileList | null) {
    if (!files || files.length === 0) return;
    const form = new FormData();
    Array.from(files).forEach((f) => form.append('files', f));
    const res = await fetch('/api/extract', { method: 'POST', body: form });
    if (!res.ok) {
      alert('Failed to extract files');
      return;
    }
    const data = (await res.json()) as { text: string };
    setCvText((prev) => (prev ? prev + '\n\n' : '') + data.text);
    setActiveTab('generate');
  }

  async function handleGenerate() {
    const key = localStorage.getItem('api_key') || '';
    const res = await fetch('/api/generate-questions', {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(key ? { 'x-api-key': key } : {}) },
      body: JSON.stringify({ cvText, jdText, numQuestions }),
    });
    const data = (await res.json()) as { questions: Question[] };
    setQuestions(data.questions || []);
    setCurrentQuestionIdx(0);
    setActiveTab('mock');
  }

  async function handleEvaluate() {
    if (!currentQuestion) return;
    const key = localStorage.getItem('api_key') || '';
    const res = await fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(key ? { 'x-api-key': key } : {}) },
      body: JSON.stringify({ question: currentQuestion.question, candidateAnswer, cvText, jdText }),
    });
    const data = (await res.json()) as { evaluation: Evaluation };
    setEvaluation(data.evaluation);
    setSuggestedAnswer(data.evaluation?.suggested_answer || '');
  }

  async function handleSuggestedAnswerStream() {
    if (!currentQuestion) return;
    setSuggestedAnswer('');
    const key = localStorage.getItem('api_key') || '';
    const res = await fetch('/api/answer', {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(key ? { 'x-api-key': key } : {}) },
      body: JSON.stringify({ question: currentQuestion.question, cvText, jdText }),
    });
    if (!res.body) return;
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const text = decoder.decode(value);
      setSuggestedAnswer((prev) => prev + text);
    }
  }

  function nextQuestion() {
    setEvaluation(null);
    setCandidateAnswer('');
    reset();
    setCurrentQuestionIdx((idx) => Math.min(idx + 1, Math.max(0, questions.length - 1)));
  }

  function prevQuestion() {
    setEvaluation(null);
    setCandidateAnswer('');
    reset();
    setCurrentQuestionIdx((idx) => Math.max(0, idx - 1));
  }

  return (
    <div className="space-y-6">
      <nav className="flex gap-2 border-b border-gray-800 pb-2">
        {(['upload', 'generate', 'mock', 'settings'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-3 py-1.5 rounded text-sm ${activeTab === tab ? 'bg-gray-800 text-white' : 'text-gray-300 hover:text-white hover:bg-gray-800/50'}`}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </nav>

      {activeTab === 'upload' && (
        <section className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <label className="text-sm text-gray-400">Upload CV (.pdf/.docx/.txt)</label>
              <input type="file" multiple accept=".pdf,.docx,.txt" className="mt-2 block w-full text-sm" onChange={(e) => handleExtract(e.target.files)} />
            </div>
            <div>
              <label className="text-sm text-gray-400">Paste JD text (optional)</label>
              <textarea className="mt-2 w-full h-32 bg-gray-900 border border-gray-800 rounded p-2 text-sm" value={jdText} onChange={(e) => setJdText(e.target.value)} />
            </div>
          </div>
          <div>
            <label className="text-sm text-gray-400">Extracted CV Text</label>
            <textarea className="mt-2 w-full h-48 bg-gray-900 border border-gray-800 rounded p-2 text-sm" value={cvText} onChange={(e) => setCvText(e.target.value)} />
          </div>
          <div className="flex justify-end">
            <button onClick={() => setActiveTab('generate')} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded">Continue</button>
          </div>
        </section>
      )}

      {activeTab === 'generate' && (
        <section className="space-y-4">
          <div className="flex items-center gap-3">
            <label className="text-sm text-gray-400">Number of questions</label>
            <input type="number" min={1} max={30} value={numQuestions} onChange={(e) => setNumQuestions(parseInt(e.target.value || '10'))} className="w-24 bg-gray-900 border border-gray-800 rounded p-1 text-sm" />
            <button onClick={handleGenerate} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded">Generate</button>
          </div>
          {questions.length > 0 && (
            <ul className="space-y-2 text-sm">
              {questions.map((q, i) => (
                <li key={q.id} className="border border-gray-800 rounded p-2">
                  <div className="text-gray-200">{i + 1}. {q.question}</div>
                  <div className="text-xs text-gray-400">{q.category} • {q.difficulty}</div>
                </li>
              ))}
            </ul>
          )}
          {questions.length > 0 && (
            <div className="flex justify-end">
              <button onClick={() => setActiveTab('mock')} className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded">Start Mock</button>
            </div>
          )}
        </section>
      )}

      {activeTab === 'mock' && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-400">Question {currentQuestionIdx + 1} / {questions.length || 0}</div>
            <div className="flex gap-2">
              <button onClick={prevQuestion} className="px-2 py-1 text-sm bg-gray-800 rounded">Prev</button>
              <button onClick={nextQuestion} className="px-2 py-1 text-sm bg-gray-800 rounded">Next</button>
            </div>
          </div>
          <div className="border border-gray-800 rounded p-3">
            <div className="text-gray-200 font-medium">{currentQuestion?.question || 'Generate questions first'}</div>
            <div className="mt-2 text-xs text-gray-400">{currentQuestion?.category} • {currentQuestion?.difficulty}</div>
            <div className="mt-3 flex gap-2">
              <button onClick={() => currentQuestion && speak(currentQuestion.question)} className="px-2 py-1 text-sm bg-gray-800 rounded">Speak</button>
              {sttSupported ? (
                listening ? (
                  <button onClick={stop} className="px-2 py-1 text-sm bg-red-700 rounded">Stop</button>
                ) : (
                  <button onClick={() => { reset(); start(); }} className="px-2 py-1 text-sm bg-green-700 rounded">Record</button>
                )
              ) : (
                <span className="text-xs text-gray-500">STT not supported in this browser</span>
              )}
            </div>
          </div>
          <div>
            <label className="text-sm text-gray-400">Your Answer</label>
            <textarea value={candidateAnswer} onChange={(e) => setCandidateAnswer(e.target.value)} className="mt-2 w-full h-40 bg-gray-900 border border-gray-800 rounded p-2 text-sm" />
          </div>
          <div className="flex gap-2">
            <button onClick={handleEvaluate} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded">Get Rating & Feedback</button>
            <button onClick={handleSuggestedAnswerStream} className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded">Generate Model Answer</button>
            <button onClick={() => suggestedAnswer && speak(suggestedAnswer)} className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded">Speak Answer</button>
          </div>
          {evaluation && (
            <div className="grid md:grid-cols-3 gap-4">
              <div className="border border-gray-800 rounded p-3">
                <div className="text-sm text-gray-400">Score</div>
                <div className="text-3xl font-bold">{evaluation.score}/10</div>
              </div>
              <div className="border border-gray-800 rounded p-3">
                <div className="text-sm text-gray-400">Strengths</div>
                <ul className="mt-2 text-sm list-disc list-inside space-y-1">
                  {evaluation.strengths?.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
              <div className="border border-gray-800 rounded p-3">
                <div className="text-sm text-gray-400">Improvements</div>
                <ul className="mt-2 text-sm list-disc list-inside space-y-1">
                  {evaluation.improvements?.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
              <div className="md:col-span-3 border border-gray-800 rounded p-3">
                <div className="text-sm text-gray-400">Summary</div>
                <p className="mt-2 text-sm">{evaluation.summary}</p>
              </div>
            </div>
          )}
          {suggestedAnswer && (
            <div className="border border-gray-800 rounded p-3">
              <div className="text-sm text-gray-400">Model Suggested Answer</div>
              <div className="prose prose-invert max-w-none mt-2 text-sm whitespace-pre-wrap">{suggestedAnswer}</div>
            </div>
          )}
        </section>
      )}

      {activeTab === 'settings' && (
        <section className="space-y-4">
          <div className="text-sm text-gray-400">Provide an API key locally to avoid any paywalls. This app uses an OpenAI-compatible API for DeepSeek R1. Keys are stored client-side only for requests you make, or use server env for self-hosting.</div>
          <KeyControls />
          <div className="text-sm text-gray-400">TTS uses your browser speechSynthesis. STT uses webkitSpeechRecognition where available.</div>
        </section>
      )}
    </div>
  );
}

function KeyControls() {
  const [key, setKey] = useState('');
  useEffect(() => {
    const saved = localStorage.getItem('api_key') || '';
    setKey(saved);
  }, []);
  function save() {
    localStorage.setItem('api_key', key);
  }
  return (
    <div className="flex items-center gap-2">
      <input type="password" placeholder="API Key (OpenRouter/DeepSeek)" value={key} onChange={(e) => setKey(e.target.value)} className="w-full bg-gray-900 border border-gray-800 rounded p-2 text-sm" />
      <button onClick={save} className="px-3 py-2 bg-gray-800 rounded">Save</button>
    </div>
  );
}
