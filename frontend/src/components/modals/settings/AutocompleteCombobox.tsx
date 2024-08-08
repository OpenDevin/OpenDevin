import { Autocomplete, AutocompleteItem, Tooltip } from "@nextui-org/react";
import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

type Label = "model" | "agent" | "language" | "securityanalyzer";

const LABELS: Record<Label, I18nKey> = {
  model: I18nKey.CONFIGURATION$MODEL_SELECT_LABEL,
  agent: I18nKey.CONFIGURATION$AGENT_SELECT_LABEL,
  language: I18nKey.CONFIGURATION$LANGUAGE_SELECT_LABEL,
  securityanalyzer: I18nKey.CONFIGURATION$SECURITY_SELECT_LABEL,
};

const PLACEHOLDERS: Record<Label, I18nKey> = {
  model: I18nKey.CONFIGURATION$MODEL_SELECT_PLACEHOLDER,
  agent: I18nKey.CONFIGURATION$AGENT_SELECT_PLACEHOLDER,
  language: I18nKey.CONFIGURATION$LANGUAGE_SELECT_PLACEHOLDER,
  securityanalyzer: I18nKey.CONFIGURATION$SECURITY_SELECT_PLACEHOLDER,
};

type AutocompleteItemType = {
  value: string;
  label: string;
};

interface AutocompleteComboboxProps {
  ariaLabel: Label;
  items: AutocompleteItemType[];
  defaultKey: string;
  onChange: (key: string) => void;
  tooltip: string;
  allowCustomValue?: boolean;
  disabled?: boolean;
}

export function AutocompleteCombobox({
  ariaLabel,
  items,
  defaultKey,
  onChange,
  tooltip,
  allowCustomValue = false,
  disabled = false,
}: AutocompleteComboboxProps) {
  const { t } = useTranslation();

  return (
    <Tooltip
      content={
        disabled
          ? `${tooltip} ${t(I18nKey.SETTINGS$DISABLED_RUNNING)}`
          : tooltip
      }
      closeDelay={100}
      delay={500}
    >
      <Autocomplete
        aria-label={ariaLabel}
        label={t(LABELS[ariaLabel])}
        placeholder={t(PLACEHOLDERS[ariaLabel])}
        defaultItems={items}
        defaultSelectedKey={defaultKey}
        inputValue={
          // Find the label for the default key, otherwise use the default key itself
          // This is useful when the default key is not in the list of items, in the case of a custom LLM model
          items.find((item) => item.value === defaultKey)?.label || defaultKey
        }
        onInputChange={(val) => {
          onChange(val);
        }}
        isDisabled={disabled}
        allowsCustomValue={allowCustomValue}
      >
        {(item) => (
          <AutocompleteItem key={item.value}>{item.label}</AutocompleteItem>
        )}
      </Autocomplete>
    </Tooltip>
  );
}
