import React, { useState, useEffect } from "react";
import { signIn, useSession, getSession } from "next-auth/react";
import Admin from "layouts/Admin.js";
import DiseaseForm from "components/DiseaseForm";
import ReactAudioPlayer from "react-audio-player";

export default function Disease() {
  const { data: session, status } = useSession();
  console.log(session);
  const [loading, setLoading] = useState(true);
  const [language, setLanguage] = useState("en-US");

  // const speak = (text) => {
  //   const utterance = new SpeechSynthesisUtterance(text);
  //   // utterance.lang = language;
  //   window.speechSynthesis.speak(utterance);
  // };
  useEffect(() => {
   setLoading(false);
  });

  if (loading) {
    return <h2 style={{ marginTop: 100, textAlign: "center" }}>LOADING...</h2>;
  }
  return (
    <>
         
    <Admin
      title="Disease Detection"
      headerText="Upload Image to detect crop disease"
      desc="Click on the play button to get the description of the page"
    >
  <ReactAudioPlayer
  src="/Disease.mp3"
  controls
/>

      <div className="flex flex-wrap mt-4 justify-center">
        <div className="w-full mb-12 xl:mb-0 px-4">
          <DiseaseForm />
        </div>
      </div>
 
    </Admin>
    </>
  );
}

export async function getServerSideProps(context) {
  const session = await getSession(context);
  let userId = null;

  return {
    props: {
      session,
      userId,
    },
  };
}
