import { motion } from "framer-motion";
import { UploadCloud } from "lucide-react";
import { useDropzone } from "react-dropzone";
import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const UploadAudioCard = () => {
  const numCircles = 6;
  const [uploadedFile, setUploadedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const generateRandomMovement = () => ({
    x: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150],
    y: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150],
  });

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setUploadedFile(acceptedFiles[0]);
      setError(""); 
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "audio/*": [] },
    maxSize: 100 * 1024 * 1024,
    noClick: true,
  });

  const handleSubmit = async () => {
    if (!uploadedFile) {
      setError("Please select an audio file first");
      return;
    }
    
    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("audio", uploadedFile);

    try {
      // Configure axios with timeout and error handling
      const res = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      });
      
      const result = res.data;
      console.log("API Response:", result);
      
      navigate("/audio-analysis-result", {
        state: {
          fileName: uploadedFile.name,
          prediction: result,
        },
      });
    } catch (error) {
      console.error("Error uploading:", error);
      
      // Handle different error types
      let errorMessage = "Prediction failed. Please try again.";
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = "Connection timed out. Please ensure the server is running.";
      } else if (error.code === 'ERR_NETWORK') {
        errorMessage = "Network error. Please check if the Flask server is running at http://localhost:5000";
      } else if (error.response) {
        // The server responded with an error status
        errorMessage = `Server error: ${error.response.data.error || error.response.statusText}`;
      }
      
      setError(errorMessage);
      
      // Still navigate but with error info
      navigate("/audio-analysis-result", {
        state: {
          fileName: uploadedFile.name,
          prediction: { error: errorMessage },
        },
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      {...getRootProps()}
      className={`relative flex flex-col items-center justify-center rounded-3xl bg-gradient-to-r from-[#0a0a0a] to-[#171717] shadow-lg overflow-hidden border-2 border-dashed ${
        isDragActive ? "border-blue-400 bg-opacity-80" : "border-gray-500"
      } p-6 transition-all duration-200`}
    >
      {/* Dropzone input (hidden visually) */}
      <input {...getInputProps()} style={{ display: "none" }} />

      {/* Background motion circles */}
      {[...Array(numCircles)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-20 h-20 bg-blue-500 opacity-25 rounded-full blur-3xl"
          animate={generateRandomMovement()}
          transition={{
            duration: 6 + Math.random() * 3,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{
            top: `${Math.random() * 100}%`,
            left: `${Math.random() * 100}%`,
          }}
        />
      ))}

      {/* Main content */}
      <div className="relative text-center h-60 w-[1400px] z-10 flex flex-col items-center">
        <h1 className="text-white text-xl font-bold">Upload Audio For Analysis</h1>
        <p className="text-gray-400 text-sm mb-4">Instantly analyze your audio using Dhwani4u</p>

        {/* Upload button */}
        <div
          className="cursor-pointer flex flex-col items-center justify-center w-56 h-20 border-2 border-blue-500 text-blue-500 rounded-lg hover:bg-blue-500 hover:text-white transition relative overflow-hidden"
          onClick={() => {
            document.getElementById("hiddenAudioInput").click();
          }}
        >
          <UploadCloud size={24} className="mb-2" />
          <span className="text-base font-medium">Choose Audio File</span>
        </div>

        {/* Hidden input outside Dropzone */}
        <input
          type="file"
          id="hiddenAudioInput"
          accept="audio/*"
          style={{ display: "none" }}
          onChange={(e) => {
            if (e.target.files?.length > 0) {
              setUploadedFile(e.target.files[0]);
              setError(""); // Clear any previous errors
            }
          }}
        />

        <p className="text-gray-400 text-xs mt-2">or, drop the file here</p>
        <p className="text-gray-500 text-xs mt-2">
          File type: MP3/WAV
        </p>

        {/* Error message */}
        {error && (
          <p className="text-red-400 text-sm mt-2">{error}</p>
        )}

        {/* File uploaded + submit */}
        {uploadedFile && (
          <div className="mt-4 flex flex-col items-center gap-3">
            <p className="text-green-400 text-sm">Uploaded: {uploadedFile.name}</p>
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition text-base"
            >
              {loading ? "Analyzing..." : "Submit"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadAudioCard

// import { motion } from "framer-motion";
// import { UploadCloud } from "lucide-react";
// import { useDropzone } from "react-dropzone";
// import { useCallback, useState } from "react";
// import { useNavigate } from "react-router-dom";
// import axios from "axios";

// const UploadAudioCard = () => {
//   const numCircles = 6;
//   const [uploadedFile, setUploadedFile] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const navigate = useNavigate();

//   const generateRandomMovement = () => ({
//     x: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150],
//     y: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150],
//   });

//   const onDrop = useCallback((acceptedFiles) => {
//     if (acceptedFiles.length > 0) {
//       setUploadedFile(acceptedFiles[0]);
//     }
//   }, []);

//   const { getRootProps, getInputProps, isDragActive } = useDropzone({
//     onDrop,
//     accept: { "audio/*": [] },
//     maxSize: 100 * 1024 * 1024,
//     noClick: true,
//   });

//   // const handleSubmit = async () => {
//   //   if (!uploadedFile) return;
//   //   setLoading(true);

//   //   const formData = new FormData();
//   //   formData.append("audio", uploadedFile);

//   //   try {
//   //     const res = await axios.post("http://localhost:5000/predict", formData);
//   //     const result = res.data;

//   //     navigate("/audio-analysis-result", {
//   //       state: {
//   //         fileName: uploadedFile.name,
//   //         prediction: result,
//   //       },
//   //     });
//   //   } catch (error) {
//   //     console.error("Error uploading:", error);
//   //     navigate("/audio-analysis-result", {
//   //       state: {
//   //         fileName: uploadedFile.name,
//   //         prediction: { error: "Prediction failed. Try again." },
//   //       },
//   //     });
//   //   } finally {
//   //     setLoading(false);
//   //   }
//   // };

// const handleSubmit = async () => {
//   if (!uploadedFile) return;
//   setLoading(true);

//   const formData = new FormData();
//   formData.append("audio", uploadedFile);

//   try {
//     // Gender and Age Prediction API
//     const genderAgeRes = await axios.post("http://localhost:5000/predict", formData);  
//     const genderAgeResult = genderAgeRes.data;

//     // Emotion Prediction API
//     const emotionRes = await axios.post("http://localhost:5000/predict-emotion", formData);  // Ensure this URL is correct
//     const emotionResult = emotionRes.data;

//     // Combine the results
//     const result = {
//       genderAge: genderAgeResult,
//       emotion: emotionResult,
//     };

//     navigate("/audio-analysis-result", {
//       state: {
//         fileName: uploadedFile.name,
//         prediction: result,
//       },
//     });
//   } catch (error) {
//     console.error("Error uploading:", error);
//     navigate("/audio-analysis-result", {
//       state: {
//         fileName: uploadedFile.name,
//         prediction: { error: "Prediction failed. Try again." },
//       },
//     });
//   } finally {
//     setLoading(false);
//   }
// };



//   return (
//     <div
//       {...getRootProps()}
//       className={`relative flex flex-col items-center justify-center rounded-3xl bg-gradient-to-r from-[#0a0a0a] to-[#171717] shadow-lg overflow-hidden border-2 border-dashed ${isDragActive ? "border-blue-400 bg-opacity-80" : "border-gray-500"
//         } p-6 transition-all duration-200`}
//     >
//       {/* Dropzone input (hidden visually) */}
//       <input {...getInputProps()} style={{ display: "none" }} />

//       {/* Background motion circles */}
//       {[...Array(numCircles)].map((_, i) => (
//         <motion.div
//           key={i}
//           className="absolute w-20 h-20 bg-blue-500 opacity-25 rounded-full blur-3xl"
//           animate={generateRandomMovement()}
//           transition={{
//             duration: 6 + Math.random() * 3,
//             repeat: Infinity,
//             ease: "easeInOut",
//           }}
//           style={{
//             top: `${Math.random() * 100}%`,
//             left: `${Math.random() * 100}%`,
//           }}
//         />
//       ))}

//       {/* Main content */}
//       <div className="relative text-center h-60 w-[1400px] z-10 flex flex-col items-center">
//         <h1 className="text-white text-xl font-bold">Upload Audio For Analysis</h1>
//         <p className="text-gray-400 text-sm mb-4">Instantly analyze your audio using Dhwani4u</p>

//         {/* Upload button */}
//         <div
//           className="cursor-pointer flex flex-col items-center justify-center w-56 h-20 border-2 border-blue-500 text-blue-500 rounded-lg hover:bg-blue-500 hover:text-white transition relative overflow-hidden"
//           onClick={() => {
//             document.getElementById("hiddenAudioInput").click();
//           }}
//         >
//           <UploadCloud size={24} className="mb-2" />
//           <span className="text-base font-medium">Choose Audio File</span>
//         </div>

//         {/* Hidden input outside Dropzone */}
//         <input
//           type="file"
//           id="hiddenAudioInput"
//           accept="audio/*"
//           style={{ display: "none" }}
//           onChange={(e) => {
//             if (e.target.files?.length > 0) {
//               setUploadedFile(e.target.files[0]);
//             }
//           }}
//         />

//         <p className="text-gray-400 text-xs mt-2">or, drop the file here</p>
//         <p className="text-gray-500 text-xs mt-2">
//           {/* Max file size: 100MB (<a className="text-blue-500 underline">Sign up</a> to increase) */}
//           File type: MP3/WAV
//         </p>

//         {/* File uploaded + submit */}
//         {uploadedFile && (
//           <div className="mt-4 flex flex-col items-center gap-3">
//             <p className="text-green-400 text-sm">Uploaded: {uploadedFile.name}</p>
//             <button
//               onClick={handleSubmit}
//               disabled={loading}
//               className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition text-base"
//             >
//               {loading ? "Analyzing..." : "Submit"}
//             </button>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// };

// export default UploadAudioCard;




// import { motion } from "framer-motion";
// import { UploadCloud } from "lucide-react";
// import { useDropzone } from "react-dropzone";
// import { useCallback, useState } from "react";
// import { useNavigate } from "react-router-dom";
// import axios from "axios";

// const UploadAudioCard = () => {
//   const numCircles = 6;
//   const [uploadedFile, setUploadedFile] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const navigate = useNavigate();

//   const generateRandomMovement = () => ({
//     x: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150],
//     y: [Math.random() * 300 - 150, Math.random() * 300 - 150, Math.random() * 300 - 150],
//   });

//   const onDrop = useCallback((acceptedFiles) => {
//     if (acceptedFiles.length > 0) {
//       setUploadedFile(acceptedFiles[0]);
//     }
//   }, []);

//   const { getRootProps, getInputProps, isDragActive } = useDropzone({
//     onDrop,
//     accept: { "audio/*": [] },
//     maxSize: 100 * 1024 * 1024,
//     noClick: true,
//   });

//   const handleSubmit = async () => {
//     if (!uploadedFile) return;
//     setLoading(true);

//     const formData = new FormData();
//     formData.append("audio", uploadedFile);

//     try {
//       const res = await axios.post("http://localhost:5000/predict", formData);
//       const result = res.data;

//       navigate("/audio-analysis-result", {
//         state: {
//           fileName: uploadedFile.name,
//           prediction: result,
//         },
//       });
//     } catch (error) {
//       console.error("Error uploading:", error);
//       navigate("/audio-analysis-result", {
//         state: {
//           fileName: uploadedFile.name,
//           prediction: { error: "Prediction failed. Try again." },
//         },
//       });
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div
//       {...getRootProps()}
//       className={`relative flex flex-col items-center justify-center rounded-3xl bg-gradient-to-r from-[#0a0a0a] to-[#171717] shadow-lg overflow-hidden border-2 border-dashed ${
//         isDragActive ? "border-blue-400 bg-opacity-80" : "border-gray-500"
//       } p-6 transition-all duration-200`}
//     >
//       {/* Dropzone input (hidden visually) */}
//       <input {...getInputProps()} style={{ display: "none" }} />

//       {/* Background motion circles */}
//       {[...Array(numCircles)].map((_, i) => (
//         <motion.div
//           key={i}
//           className="absolute w-20 h-20 bg-blue-500 opacity-25 rounded-full blur-3xl"
//           animate={generateRandomMovement()}
//           transition={{
//             duration: 6 + Math.random() * 3,
//             repeat: Infinity,
//             ease: "easeInOut",
//           }}
//           style={{
//             top: `${Math.random() * 100}%`,
//             left: `${Math.random() * 100}%`,
//           }}
//         />
//       ))}

//       {/* Main content */}
//       <div className="relative text-center h-60 w-[1400px] z-10 flex flex-col items-center">
//         <h1 className="text-white text-xl font-bold">Upload Audio For Analysis</h1>
//         <p className="text-gray-400 text-sm mb-4">Instantly analyze your audio using Dhwani4u</p>

//         {/* Upload button */}
//         <div
//           className="cursor-pointer flex flex-col items-center justify-center w-56 h-20 border-2 border-blue-500 text-blue-500 rounded-lg hover:bg-blue-500 hover:text-white transition relative overflow-hidden"
//           onClick={() => {
//             document.getElementById("hiddenAudioInput").click();
//           }}
//         >
//           <UploadCloud size={24} className="mb-2" />
//           <span className="text-base font-medium">Choose Audio File</span>
//         </div>

//         {/* Hidden input outside Dropzone */}
//         <input
//           type="file"
//           id="hiddenAudioInput"
//           accept="audio/*"
//           style={{ display: "none" }}
//           onChange={(e) => {
//             if (e.target.files?.length > 0) {
//               setUploadedFile(e.target.files[0]);
//             }
//           }}
//         />

//         <p className="text-gray-400 text-xs mt-2">or, drop the file here</p>
//         <p className="text-gray-500 text-xs mt-2">
//           {/* Max file size: 100MB (<a className="text-blue-500 underline">Sign up</a> to increase) */}
//           File type: MP3/WAV
//         </p>

//         {/* File uploaded + submit */}
//         {uploadedFile && (
//           <div className="mt-4 flex flex-col items-center gap-3">
//             <p className="text-green-400 text-sm">Uploaded: {uploadedFile.name}</p>
//             <button
//               onClick={handleSubmit}
//               disabled={loading}
//               className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition text-base"
//             >
//               {loading ? "Analyzing..." : "Submit"}
//             </button>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// };

// export default UploadAudioCard;



