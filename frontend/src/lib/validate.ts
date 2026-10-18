export const ACCEPTED_EXT = [".mov", ".mp4"];
export const ACCEPTED_MIME = ["video/quicktime", "video/mp4", "video/x-m4v", ""];

function envMaxMb(): number {
  const raw = (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_MAX_UPLOAD_MB;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : 1024;
}

export const MAX_UPLOAD_MB = envMaxMb();

export interface FileLike {
  name: string;
  size: number;
  type: string;
}

/** null — файл подходит; иначе текст ошибки для пользователя. */
export function validateVideoFile(f: FileLike, maxMb = MAX_UPLOAD_MB): string | null {
  const name = f.name.toLowerCase();
  const okExt = ACCEPTED_EXT.some((e) => name.endsWith(e));
  if (!okExt) return `Формат «${f.name.split(".").pop() ?? "?"}» не поддерживается. Нужен .mov или .mp4.`;
  if (f.type && !ACCEPTED_MIME.includes(f.type) && !f.type.startsWith("video/"))
    return "Файл не похож на видеозапись.";
  if (f.size <= 0) return "Файл пустой.";
  if (f.size > maxMb * 1024 * 1024) return `Файл больше ${maxMb} МБ — загрузите запись меньшего размера.`;
  return null;
}
