interface UploadedFilePreviewProps {
  file: string; // base64
  onRemove: () => void;
}

export function UploadedFilePreview({
  file,
  onRemove,
}: UploadedFilePreviewProps) {
  return (
    <div className="relative flex-shrink-0">
      <button
        type="button"
        aria-label="Remove"
        onClick={onRemove}
        className="absolute right-1 top-1 text-[#A3A3A3] hover:text-danger"
      >
        &times;
      </button>
      <img src={file} alt="" className="w-16 h-16 aspect-auto rounded" />
    </div>
  );
}
