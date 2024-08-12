import React, { createContext, useContext, useMemo, useState } from "react";

export interface UnsavedEdit {
  path: string;
  content: string;
}

interface UnsavedEdits {
  unsavedEdits: UnsavedEdit[];
  setUnsavedEdits: (edits: UnsavedEdit[]) => void;
  addOrUpdateUnsavedEdit: (path: string, content: string) => void;
  removeUnsavedEdit: (path: string) => void;
  hasUnsavedEdit: (path: string) => boolean;
}

const UnsavedEditsProviderContext = createContext<UnsavedEdits | null>(null);

interface UnsavedEditsProviderProperties {
  children: JSX.Element | JSX.Element[];
}

export function UnsavedEditsProvider({
  children,
}: UnsavedEditsProviderProperties) {
  const [unsavedEdits, setUnsavedEdits] = useState<UnsavedEdit[]>([]);

  const addOrUpdateUnsavedEdit = (path: string, content: string) => {
    const newUnsavedEdits = unsavedEdits.filter(
      (unsavedEdit) => unsavedEdit.path !== path,
    );
    newUnsavedEdits.push({ path, content });
    setUnsavedEdits(newUnsavedEdits);
  };

  const removeUnsavedEdit = (path: string) => {
    const newUnsavedEdits = unsavedEdits.filter(
      (unsavedEdit) => unsavedEdit.path !== path,
    );
    setUnsavedEdits(newUnsavedEdits);
  };

  const hasUnsavedEdit = (path: string) =>
    !!unsavedEdits.find((unsavedEdit) => unsavedEdit.path === path);

  const state = useMemo<UnsavedEdits>(
    () => ({
      unsavedEdits,
      setUnsavedEdits,
      addOrUpdateUnsavedEdit,
      removeUnsavedEdit,
      hasUnsavedEdit,
    }),
    [unsavedEdits],
  );

  return (
    <UnsavedEditsProviderContext.Provider value={state}>
      {children}
    </UnsavedEditsProviderContext.Provider>
  );
}

export const useUnsavedEdits = () =>
  useContext(UnsavedEditsProviderContext) as unknown as UnsavedEdits;
