import React from "react";

interface SuggestionBoxProps {
  title: string;
  content: React.ReactNode;
}

export function SuggestionBox({ title, content }: SuggestionBoxProps) {
  return (
    <div className="w-full h-[100px] px-4 border border-[#525252] rounded-xl flex flex-col items-center justify-center gap-1">
      <span className="text-[16px] leading-6 -tracking-[0.01em] font-[600]">
        {title}
      </span>
      {typeof content === "string" ? (
        <span className="text-sm">{content}</span>
      ) : (
        <div className="w-full">{content}</div>
      )}
    </div>
  );
}
