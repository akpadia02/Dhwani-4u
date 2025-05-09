import { motion } from "framer-motion";
import { useState } from "react";
import { UploadCloud } from "lucide-react";

const Info = () => {
  const [flipped, setFlipped] = useState(false);
  const numCircles = 6; // Floating background elements

  const generateRandomMovement = () => ({
    x: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150],
    y: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150]
  });

  return (
    <div 
      className="relative flex items-center justify-center w-full h-[350px] md:h-60 rounded-3xl bg-gradient-to-r from-[#0a0a0a] to-[#171717] shadow-lg overflow-hidden cursor-pointer"
      onClick={() => setFlipped(!flipped)}
    >
      {[...Array(numCircles)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-40 h-40 bg-blue-500 opacity-25 rounded-full blur-3xl"
          animate={generateRandomMovement()}
          transition={{ duration: 6 + Math.random() * 3, repeat: Infinity, ease: "easeInOut" }}
          style={{ top: `${Math.random() * 100}%`, left: `${Math.random() * 100}%` }}
        />
      ))}

      <motion.div 
        className="relative text-center z-10 flex flex-col items-center w-full h-full p-6 text-white rounded-3xl"
        animate={{ rotateY: flipped ? 180 : 0 }}
        transition={{ duration: 0.6 }}
        style={{ transformStyle: "preserve-3d" }}
      >
        <div className={`absolute w-full h-full backface-hidden ${flipped ? "hidden" : "flex flex-col items-center"}`}>
          <h1 className="text-2xl font-bold">What We Provide?</h1>
          <p className="text-lg  mt-4 md:mt-2">Analyzes audio in real-time to extract Age, Gender, Emotion and Intent with high accuracy.</p>
          <ul className="mt-4 text-sm text-gray-300">
            <li><strong>Age & Gender –</strong> Identifies speaker demographics.</li>
            <li><strong>Emotion Analysis –</strong> Detects moods like happiness, anger, or sadness.</li>
            <li><strong>Intent Recognition –</strong> Understands purpose.</li>
          </ul>
          <button className="mt-4 px-4 py-2 bg-blue-500 rounded-lg hover:bg-blue-600" onClick={(e) => { e.stopPropagation(); setFlipped(true); }}>Read More</button>
        </div>
        
        <div className={`absolute w-full h-full backface-hidden ${flipped ? "flex flex-col items-center" : "hidden"}`} style={{ transform: "rotateY(180deg)" }}>
          <h1 className="text-2xl font-bold">Detailed Features</h1>
          <ul className="mt-4 text-base ml-4 md:ml-0 text-gray-300 text-left">
            <li>✅ <strong>Real-time AI Analysis –</strong> Fast & accurate voice processing</li>
            <li>✅ <strong>Multi-Language Support –</strong> Works across various languages</li>
            <li>✅ <strong>Emotion & Intent Detection –</strong> Understand speaker’s mood and intent</li>
            <li>✅ <strong>Versatile Applications –</strong> Customer support, surveillance, accessibility, etc.</li>
          </ul>
          <button className="mt-4 px-4 py-2 bg-blue-500 rounded-lg hover:bg-blue-600" onClick={(e) => { e.stopPropagation(); setFlipped(false); }}>Back</button>
        </div>
      </motion.div>
    </div>
  );
};

export default Info;