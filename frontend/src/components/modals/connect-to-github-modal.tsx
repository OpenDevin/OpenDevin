import { useFetcher, useRouteLoaderData } from "@remix-run/react";
import React from "react";
import ModalBody from "./ModalBody";
import { CustomInput } from "../form/custom-input";
import ModalButton from "../buttons/ModalButton";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "./confirmation-modals/BaseModal";
import { clientLoader } from "#/root";
import { clientAction } from "#/routes/login";

interface ConnectToGitHubModalProps {
  onClose: () => void;
}

export function ConnectToGitHubModal({ onClose }: ConnectToGitHubModalProps) {
  const data = useRouteLoaderData<typeof clientLoader>("root");
  const fetcher = useFetcher<typeof clientAction>({ key: "login" });

  return (
    <ModalBody>
      <div className="flex flex-col gap-2 self-start">
        <BaseModalTitle title="Connect to GitHub" />
        <BaseModalDescription
          description={
            <span>
              Get your token{" "}
              <a
                href="https://github.com/settings/tokens/new?description=openhands-app&scopes=repo,user"
                target="_blank"
                rel="noreferrer noopener"
                className="text-[#791B80] underline"
              >
                here
              </a>
            </span>
          }
        />
      </div>
      <fetcher.Form
        method="POST"
        action="/login"
        className="w-full flex flex-col gap-6"
        onSubmit={onClose}
      >
        <CustomInput
          label="GitHub Token"
          name="ghToken"
          required
          defaultValue={data?.ghToken ?? ""}
        />

        <div className="flex flex-col gap-2 w-full">
          <ModalButton
            type="submit"
            text="Connect"
            disabled={fetcher.state === "submitting"}
            className="bg-[#791B80] w-full"
          />
          <ModalButton
            onClick={onClose}
            text="Close"
            className="bg-[#737373] w-full"
          />
        </div>
      </fetcher.Form>
    </ModalBody>
  );
}