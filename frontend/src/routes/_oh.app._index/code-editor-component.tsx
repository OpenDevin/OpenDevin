import { Editor, Monaco } from "@monaco-editor/react";
import React from "react";
import { useTranslation } from "react-i18next";
import { VscCode } from "react-icons/vsc";
import { type editor } from "monaco-editor";
import { I18nKey } from "#/i18n/declaration";
import { useFiles } from "#/context/files";
import OpenHands from "#/api/open-hands";

interface CodeEditorCompoonentProps {
  isReadOnly: boolean;
}

function CodeEditorCompoonent({ isReadOnly }: CodeEditorCompoonentProps) {
  const { t } = useTranslation();
  const {
    files,
    selectedPath,
    modifiedFiles,
    modifyFileContent,
    saveFileContent: saveNewFileContent,
  } = useFiles();

  const handleEditorDidMount = React.useCallback(
    (editor: editor.IStandaloneCodeEditor, monaco: Monaco): void => {
      monaco.editor.defineTheme("my-theme", {
        base: "vs-dark",
        inherit: true,
        rules: [],
        colors: {
          "editor.background": "#171717",
        },
      });

      monaco.editor.setTheme("my-theme");
    },
    [],
  );

  const handleEditorChange = (value: string | undefined) => {
    if (selectedPath && value) modifyFileContent(selectedPath, value);
  };

  React.useEffect(() => {
    const handleSave = async (event: KeyboardEvent) => {
      if (selectedPath && event.metaKey && event.key === "s") {
        const content = saveNewFileContent(selectedPath);

        if (content) {
          try {
            const token = localStorage.getItem("token")?.toString();
            if (token) await OpenHands.saveFile(token, selectedPath, content);
          } catch (error) {
            // handle error
          }
        }
      }
    };

    document.addEventListener("keydown", handleSave);
    return () => {
      document.removeEventListener("keydown", handleSave);
    };
  }, [saveNewFileContent]);

  if (!selectedPath) {
    return (
      <div
        data-testid="code-editor-empty-message"
        className="flex flex-col items-center text-neutral-400"
      >
        <VscCode size={100} />
        {t(I18nKey.CODE_EDITOR$EMPTY_MESSAGE)}
      </div>
    );
  }

  return (
    <Editor
      data-testid="code-editor"
      height="100%"
      path={selectedPath ?? undefined}
      defaultValue=""
      value={
        selectedPath
          ? modifiedFiles[selectedPath] || files[selectedPath]
          : undefined
      }
      onMount={handleEditorDidMount}
      onChange={handleEditorChange}
      options={{ readOnly: isReadOnly }}
    />
  );
}

export default React.memo(CodeEditorCompoonent);
