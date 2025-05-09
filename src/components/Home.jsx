import videobg from '../assets/video.mp4';
const Home = () => {
    return (
        //   <div className="bg-[#0F1D55] rounded-3xl p-6 flex justify-center items-center w-full h-[400px]">\
        //     <video src={videobg} autoPlay loop muted/>
        //     <h1 className="text-white text-4xl font-bold">Dhwani4U</h1>
        //   </div>
        <div className='main rounded-3xl flex justify-center items-center'>
            <video src={videobg} autoPlay loop muted />
            <div className='content'>
                <h1 className='text-3xl md:text-6xl font-bold'>Dhwani4U</h1>
                {/* <p className="text-white text-sm md:text-2xl mt-6 ml-12 md:ml-0 mr-12 md:mr-0">
                    Dhwani4U is an AI-powered audio analysis system that extracts age, gender, intent, and emotion from voice recordings.
                </p> */}
            </div>
        </div>
    );
};

export default Home;
