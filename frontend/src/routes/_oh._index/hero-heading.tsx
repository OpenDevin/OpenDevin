import BuildIt from "#/assets/build-it.svg?react";

export function HeroHeading() {
  return (
    <div className="w-[304px] text-center flex flex-col gap-4 items-center py-4">
      <BuildIt width={88} height={104} />
      <h1 className="text-[38px] leading-[32px] -tracking-[0.02em]">
        Let&apos;s Start Building!
      </h1>
      <p className="mx-4 text-sm flex flex-col gap-2">
        OpenHands makes it easy to build and maintain software using a simple
        prompt.{" "}
        <span className="">
          Not sure how to start?{" "}
          <a
            rel="noopener noreferrer"
            target="_blank"
            href="https://docs.all-hands.dev/modules/usage/getting-started"
            className="text-hyperlink underline underline-offset-[3px]"
          >
            Read this
          </a>
        </span>
      </p>
    </div>
  );
}
