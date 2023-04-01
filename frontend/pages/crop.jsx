import React, { useState, useEffect } from "react";
import { signIn, useSession, getSession } from "next-auth/react";
import Admin from "layouts/Admin.js";
import CropForm from "components/CropForm";
import ReactAudioPlayer from "react-audio-player";

export default function Crop() {
  const { data: session, status } = useSession();
  console.log(session);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
   setLoading(false);
  });

  if (loading) {
    return <h2 style={{ marginTop: 100, textAlign: "center" }}>LOADING...</h2>;
  }
  return (
    <Admin
      title="Crop Recommendation"
      headerText="Enter details to get crop recommendations"
      desc="Click on the play button to get the description of the page"
    >
        <ReactAudioPlayer
  src="/Crop_rec.mp3"
  controls
/>
      <div className="flex flex-wrap mt-4 justify-center">
        <div className="w-full mb-12 xl:mb-0 px-4">
          <CropForm />
        </div>
      </div>
    </Admin>
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
