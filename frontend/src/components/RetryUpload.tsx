import { useState } from "react";
import { RotateCcw } from "lucide-react";
import { ApiError, localFile, rememberLocalVideo, saveHistoryItem } from "../api/client";
import { useApp } from "../context";
import { navigate } from "../lib/router";

/** Повторная отправка того же файла (если он ещё в памяти вкладки) — новая задача. */
export default function RetryUpload({ jobId, className }: { jobId: string; className?: string }) {
  const { backend, toast } = useApp();
  const file = localFile(jobId);
  const [p, setP] = useState<number | null>(null);
  if (!file || backend.kind === "demo") return null;
  return (
    <button
      className={className ?? "btn-ghost"}
      disabled={p !== null}
      onClick={async () => {
        setP(0);
        try {
          const id = await backend.upload(file, setP).promise;
          rememberLocalVideo(id, file);
          saveHistoryItem({ job_id: id, file_name: file.name, size: file.size, status: "processing" });
          navigate({ name: "job", id });
        } catch (e) {
          toast(e instanceof ApiError ? e.message : "Не удалось отправить файл повторно", "error");
        } finally {
          setP(null);
        }
      }}
    >
      <RotateCcw className="h-4 w-4" />
      {p === null ? "Отправить этот файл повторно" : `Отправка… ${Math.round(p * 100)} %`}
    </button>
  );
}
