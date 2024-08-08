import React from "react";
import SecurityInvariant from "./invariant/Invariant";
import BaseModal from "../base-modal/BaseModal";
import { getSettings } from "#/services/settings";

interface SecurityProps {
  isOpen: boolean;
  onOpenChange: (isOpen: boolean) => void;
}

enum SecurityAnalyzerOption {
  INVARIANT = "invariant",
}

const SecurityAnalyzers: Record<SecurityAnalyzerOption, React.ElementType> = {
  [SecurityAnalyzerOption.INVARIANT]: SecurityInvariant,
};

function Security({ isOpen, onOpenChange }: SecurityProps): JSX.Element {
  const { SECURITY_ANALYZER } = getSettings();
  const AnalyzerComponent =
    SECURITY_ANALYZER &&
    SecurityAnalyzers[SECURITY_ANALYZER as SecurityAnalyzerOption]
      ? SecurityAnalyzers[SECURITY_ANALYZER as SecurityAnalyzerOption]
      : () => <div>Unknown security analyzer chosen</div>;

  return (
    <BaseModal
      isOpen={isOpen && !!SECURITY_ANALYZER}
      contentClassName="max-w-[80%] h-[80%]"
      bodyClassName="px-0 py-0 max-h-[100%]"
      onOpenChange={onOpenChange}
      title=""
    >
      <AnalyzerComponent />
    </BaseModal>
  );
}

export default Security;
