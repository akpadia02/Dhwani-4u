import React, { useState } from 'react';
import { motion } from "framer-motion";

function Result2() {
  const [text, setText] = useState('');
  const [isListening, setIsListening] = useState(false);

  const handleStartListening = () => {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.start();

    recognition.onresult = (event) => {
      const speechToText = event.results[0][0].transcript;
      setText(speechToText);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    setIsListening(true);
  };

  const handleTextToAudio = () => {
    const speechSynthesis = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    speechSynthesis.speak(utterance);
  };

  const numCircles = 10;

  const generateRandomMovement = () => ({
    x: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150],
    y: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150],
  });

  return (
    <div className="relative flex items-center justify-center w-full min-h-[300px] rounded-3xl bg-gradient-to-r from-[#0a0a0a] to-[#171717] shadow-lg overflow-hidden cursor-default p-6">
      {[...Array(numCircles)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-40 h-40 bg-blue-500 opacity-25 rounded-full blur-3xl"
          animate={generateRandomMovement()}
          transition={{ duration: 6 + Math.random() * 3, repeat: Infinity, ease: "easeInOut" }}
          style={{ top: `${Math.random() * 100}%`, left: `${Math.random() * 100}%` }}
        />
      ))}
      <div className="relative z-10 text-white text-center w-full max-w-5xl mx-auto">
        <h1 className="text-white text-xl font-bold mb-5">Audio to Text and Text To Speech</h1>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Speech-to-text result will appear here."
          className="w-full h-24 bg-gray-700 text-white p-4 rounded-md focus:outline-none"
        />

        <div className="flex justify-center gap-4 mt-4">
          <button
            onClick={handleStartListening}
            disabled={isListening}
            className="bg-blue-600 text-white py-2 px-4 rounded-full hover:bg-blue-700 focus:outline-none"
          >
            {isListening ? 'Listening...' : 'Start Listening'}
          </button>
          <button
            onClick={handleTextToAudio}
            disabled={!text}
            className="bg-blue-600 text-white py-2 px-4 rounded-full hover:bg-blue-700 focus:outline-none"
          >
            Speak Text
          </button>
        </div>
      </div>
    </div>

  );
}

export default Result2;
