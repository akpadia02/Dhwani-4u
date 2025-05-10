import { motion } from "framer-motion";
import { useLocation, useNavigate } from "react-router-dom";
import { useEffect } from "react";
import { Smile, User, Mic, Target, AudioLines,Calendar1 } from "lucide-react";
import { detectIntentFromText } from "./gemini";

const AudioAnalysisResult = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { fileName, prediction } = location.state || {};

  // useEffect(() => {
  //   if (!fileName || !prediction) navigate("/");
  // }, [fileName, prediction, navigate]);

  useEffect(() => {
    if (!fileName || !prediction) {
      navigate("/");
    } else if (!prediction.intent) {
      prediction.intent = "Unknown"; // fallback
    }
  }, [fileName, prediction, navigate]);

  const numCircles = 6;

  const generateRandomMovement = () => ({
    x: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150],
    y: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150],
  });

  // Function to handle Text-to-Speech
  const speakText = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    speechSynthesis.speak(utterance);
  };

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
        <h2 className="text-3xl font-bold mb-2">Here is the analysis of your audio!</h2>
        <p className="text-sm text-gray-400 mb-4 max-w-xl mx-auto">
          We've carefully analyzed your uploaded file to detect the speaker's gender, emotional tone, intent behind the speech, and estimated age group. Below are the AI-derived results:
        </p>
        <p className="text-sm text-gray-400 mb-6">Analyzed File: <span className="text-white font-medium">{fileName}</span></p>

        {prediction?.error ? (
          <p className="text-red-400 text-lg font-semibold">{prediction.error}</p>
        ) : (
          <div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 w-full">
              <div className="flex flex-col items-center">
                <User className="text-blue-400 mb-1" size={28} />
                <p className="text-sm text-gray-300">Gender</p>
                <p className="font-semibold">{prediction.gender}</p>
              </div>
              <div className="flex flex-col items-center">
                <Smile className="text-blue-400 mb-1" size={28} />
                <p className="text-sm text-gray-300">Emotion</p>
                <p className="font-semibold">{prediction.emotion || "N/A"}</p>
              </div>
              <div className="flex flex-col items-center">
                <Target className="text-blue-400 mb-1" size={28} />
                <p className="text-sm text-gray-300">Intent</p>
                <p className="font-semibold">{prediction.intent || "N/A"}</p>
              </div>
              <div className="flex flex-col items-center">
                <Calendar1 className="text-blue-400 mb-1" size={28} />
                <p className="text-sm text-gray-300">Age</p>
                <p className="font-semibold">{prediction.age}</p>
              </div>
            </div>
            <div className="flex flex-col items-center mt-10">
              <AudioLines className="text-blue-400 mb-1" size={28} />
              <p className="text-sm text-gray-300">Transcribed Text</p>
              <p className="font-semibold">{prediction.text || "N/A"}</p>
              {prediction.text && (
                <button
                  onClick={() => speakText(prediction.text)}
                  className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition"
                >
                  Speak Text
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AudioAnalysisResult;
