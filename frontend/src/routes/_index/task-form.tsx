import React from "react";
import { Form, useNavigation } from "@remix-run/react";
import Send from "#/assets/send.svg?react";
import Clip from "#/assets/clip.svg?react";
import { cn } from "#/utils/utils";

export function TaskForm() {
  const navigation = useNavigation();

  const formRef = React.useRef<HTMLFormElement>(null);
  const [hasText, setHasText] = React.useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setHasText(!!e.target.value);
  };

  return (
    <div className="flex flex-col gap-2">
      <Form ref={formRef} method="post" className="relative">
        <input
          name="q"
          type="text"
          placeholder="What do you want to build?"
          onChange={handleChange}
          className={cn(
            "bg-[#404040] placeholder:text-[#A3A3A3] border border-[#525252] w-[600px] rounded-lg px-[16px] py-[18px] text-[17px] leading-5",
            "focus:bg-[#525252]",
          )}
        />
        {hasText && (
          <button
            type="submit"
            aria-label="Submit"
            className="absolute right-4 top-1/2 transform -translate-y-1/2"
            disabled={navigation.state === "loading"}
          >
            <Send width={24} height={24} />
          </button>
        )}
      </Form>
      <button
        type="button"
        className="flex self-start items-center text-[#A3A3A3] text-xs leading-[18px] -tracking-[0.08px]"
      >
        <Clip width={16} height={16} />
        Attach a file
      </button>
    </div>
  );
}