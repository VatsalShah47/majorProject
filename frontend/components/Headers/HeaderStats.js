import React from "react";
import ReactAudioPlayer from 'react-audio-player';

// components

import CardStats from "components/Cards/CardStats.js";

export default function HeaderStats({ headerText ,desc }) {
  return (
    <>
      {/* Header */}
      
      <div className="relative bg-blueGray-800 md:pt-32 pb-32 pt-12">


        <div className="text-4xl text-white text-center font-semibold">
          {headerText}

        </div>
        <div className="text-4m text-white text-center font-semibold">
          {desc}
                </div>
      </div>
    </>
  );
}
