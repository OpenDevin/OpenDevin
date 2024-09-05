import React from "react";
import { IoMdVolumeHigh, IoMdVolumeOff } from "react-icons/io";
import beep from "#/utils/beep";

// TODO: Set prefs in cookie
function VolumeIcon(): JSX.Element {
  const [isMuted, setIsMuted] = React.useState(true);

  const toggleMute = () => {
    setIsMuted(!isMuted);
    if (!isMuted) {
      // document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;`;
    } else {
      // document.cookie = `${cookieName}=on;`;
      beep();
    }
  };

  return (
    <div
      className="cursor-pointer hover:opacity-80 transition-all"
      onClick={toggleMute}
    >
      {isMuted ? <IoMdVolumeOff size={23} /> : <IoMdVolumeHigh size={23} />}
    </div>
  );
}

export default VolumeIcon;
