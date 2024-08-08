import React from "react";

interface InvariantLogoIconProps {
  className?: string;
}

function InvariantLogoIcon({ className }: InvariantLogoIconProps): JSX.Element {
  return (
    <svg
      width="39"
      height="39"
      viewBox="0 0 39 39"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <mask
        id="mask0_6001_732"
        style={{ maskType: "alpha" }}
        maskUnits="userSpaceOnUse"
        x="0"
        y="0"
        width="39"
        height="39"
      >
        <rect width="38.9711" height="39" rx="1.90143" fill="black" />
      </mask>
      <g mask="url(#mask0_6001_732)">
        <rect
          width="38.9711"
          height="39"
          rx="4.96091"
          fill="url(#paint0_linear_6001_732)"
        />
      </g>
      <g clipPath="url(#clip0_6001_732)">
        <path
          fillRule="evenodd"
          clipRule="evenodd"
          d="M30.6946 22.9468L24.6617 19.3906C23.0017 18.412 21.9826 16.6281 21.9826 14.7005V7.64124C21.9826 6.24917 20.8546 5.12061 19.4631 5.12061H19.2448C17.8533 5.12061 16.7253 6.24917 16.7253 7.64124V14.6683C16.7253 16.5959 15.7062 18.3799 14.0461 19.3584L7.95872 22.9468C6.70795 23.6841 6.29135 25.2963 7.02841 26.5476C7.76534 27.7989 9.37687 28.2157 10.6276 27.4783L16.5643 23.9788C18.269 22.9739 20.3843 22.9739 22.089 23.9788L28.0256 27.4783C29.2764 28.2155 30.8878 27.7989 31.6249 26.5476C32.3618 25.2963 31.9453 23.6842 30.6946 22.9468ZM10.6709 11.2274L13.5534 12.9268C14.8042 13.6641 15.2206 15.2762 14.4836 16.5275L14.4835 16.5276C13.7464 17.7789 12.135 18.1955 10.8843 17.4581L8.0018 15.7588C6.75106 15.0215 6.33462 13.4094 7.07166 12.1581L7.07173 12.158C7.80876 10.9067 9.42018 10.4901 10.6709 11.2274ZM30.6885 15.7597L27.806 17.459C26.5552 18.1963 24.9438 17.7797 24.2068 16.5284L24.2067 16.5283C23.4697 15.277 23.8861 13.6649 25.1368 12.9276L28.0193 11.2283C29.2701 10.4909 30.8815 10.9075 31.6185 12.1588L31.6186 12.1589C32.3556 13.4102 31.9392 15.0223 30.6885 15.7597ZM21.9766 27.6046V30.9518C21.9766 32.4042 20.7997 33.5815 19.3479 33.5815H19.3478C17.8961 33.5815 16.7192 32.4042 16.7192 30.9518V27.6046C16.7192 26.1522 17.8961 24.9749 19.3478 24.9749H19.3479C20.7997 24.9749 21.9766 26.1522 21.9766 27.6046Z"
          fill="url(#paint1_linear_6001_732)"
        />
      </g>
      <defs>
        <linearGradient
          id="paint0_linear_6001_732"
          x1="0"
          y1="0"
          x2="39.1786"
          y2="39.1496"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#6360FD" />
          <stop offset="1" stopColor="#4541EC" />
        </linearGradient>
        <linearGradient
          id="paint1_linear_6001_732"
          x1="32.1372"
          y1="33.5815"
          x2="7.91553"
          y2="6.29303"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#DDDDDD" />
          <stop offset="1" stopColor="white" />
        </linearGradient>
        <clipPath id="clip0_6001_732">
          <rect
            width="28.4724"
            height="28.4936"
            fill="white"
            transform="translate(5.08594 5.08813)"
          />
        </clipPath>
      </defs>
    </svg>
  );
}

export default InvariantLogoIcon;
