// import { Route, Routes } from "react-router-dom";
// import Card from "./components/Card";
// import Home from "./components/Home";
// import AudioAnalysisResult from "./components/Result";



// const App = () => {
//   return (
//     <Routes >
//       <div className="bg-black min-h-screen flex flex-col items-center p-10 gap-6">
//         <Home />
//         <div className="flex flex-col md:flex-row gap-6 w-full mt-11">
//           {/* <Card /> */}
//           {/* <Info /> */}
//           <Route path="/" element={<Card />} />
//           <Route path="/audio-analysis-result" element={<AudioAnalysisResult />} />
//         </div>
//       </div>
//     </Routes>

//   );
// };

// export default App;

import { Route, Routes } from "react-router-dom";
import Card from "./components/Card";
import Home from "./components/Home";
import AudioAnalysisResult from "./components/Result";
import Result2 from "./components/Result2";

const App = () => {
  return (
    <div className="bg-black min-h-screen flex flex-col items-center p-10 gap-6">
      <Home />
      <div className="flex flex-col md:flex-row gap-6 w-full mt-11">
        <Routes>
          <Route path="/" element={<Card />} />
          <Route path="/audio-analysis-result" element={<AudioAnalysisResult />} />
        </Routes>
      </div>
      <Result2 />
    </div>
  );
};

export default App;
