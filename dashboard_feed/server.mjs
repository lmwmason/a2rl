import { createHash, verify } from "node:crypto";
import { mkdir, open, readFile, readdir, rename, unlink, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

const origin = "https://a2rl-fpv-dashboard.front.cbgpu.com";
const directory = process.env.A2RL_FEED_DIR || join(tmpdir(), "a2rl-fpv-feed");
const publicKey = await readFile(new URL("./publisher.pem", import.meta.url));
const videoName = /^replay-[0-9]{1,24}\.mp4$/;
const lastWrite = new Map();
await mkdir(directory, { recursive: true });

function reply(response, code, body = "") {
  response.writeHead(code, { "Content-Type": "text/plain; charset=utf-8" });
  response.end(body);
}

async function put(request, response, name) {
  const isStatus = name === "status.json";
  const type = isStatus ? "status" : "replay";
  const timestamp = request.headers["x-a2rl-time"] || "";
  const digest = request.headers["x-a2rl-sha256"] || "";
  const signature = request.headers["x-a2rl-signature"] || "";
  const encodedMeta = request.headers["x-a2rl-meta"] || "";
  const message = Buffer.from([name, timestamp, digest, encodedMeta].join("\n"));
  if (!/^\d{13}$/.test(timestamp) || Math.abs(Date.now() - Number(timestamp)) > 120_000 ||
      !/^[a-f0-9]{64}$/.test(digest) || !/^[A-Za-z0-9+/]{86}==$/.test(signature) ||
      encodedMeta.length > 2048 || !verify(null, message, publicKey, Buffer.from(signature, "base64"))) {
    return reply(response, 401, "Invalid publisher signature");
  }
  if (Number(timestamp) <= (lastWrite.get(type) || 0)) return reply(response, 409, "Older update");

  const limit = isStatus ? 16_384 : 32 * 1024 * 1024;
  const length = Number(request.headers["content-length"]);
  if (!Number.isSafeInteger(length) || length < 1 || length > limit) return reply(response, 413);
  const chunks = [];
  let received = 0;
  for await (const chunk of request) {
    received += chunk.length;
    if (received > length) return reply(response, 413);
    chunks.push(chunk);
  }
  const body = Buffer.concat(chunks, received);
  if (received !== length || createHash("sha256").update(body).digest("hex") !== digest) {
    return reply(response, 400, "Content hash mismatch");
  }
  if (Number(timestamp) <= (lastWrite.get(type) || 0)) return reply(response, 409, "Older update");

  let manifest;
  try {
    if (isStatus) {
      JSON.parse(body.toString("utf8"));
    } else {
      manifest = JSON.parse(Buffer.from(encodedMeta, "base64").toString("utf8"));
      if (manifest.url !== name || name !== `replay-${manifest.version}.mp4` ||
          body.subarray(4, 8).toString() !== "ftyp") return reply(response, 400);
    }
  } catch {
    return reply(response, 400, "Invalid data");
  }
  lastWrite.set(type, Number(timestamp));
  await publish(name, body, timestamp);
  if (manifest) {
    await publish("replay.json", JSON.stringify(manifest), timestamp);
    const videos = (await readdir(directory)).filter((file) => videoName.test(file) && file !== name);
    videos.sort((a, b) => b.localeCompare(a, undefined, { numeric: true }));
    await Promise.all(videos.slice(2).map((file) => unlink(join(directory, file)).catch(() => {})));
  }
  return reply(response, 204);
}

async function publish(name, value, timestamp) {
  const temp = join(directory, `.${name}.${timestamp}`);
  await writeFile(temp, value);
  await rename(temp, join(directory, name));
}

async function handle(request, response) {
  const name = new URL(request.url, "http://localhost").pathname.slice(1);
  response.setHeader("Vary", "Origin");
  response.setHeader("Cache-Control", "no-store");
  if (request.headers.origin === origin) response.setHeader("Access-Control-Allow-Origin", origin);
  if (name === "health" && request.method === "GET") return reply(response, 200, "ok");

  const isVideo = videoName.test(name);
  if (name !== "status.json" && name !== "replay.json" && !isVideo) return reply(response, 404);
  if (request.method === "PUT" && name !== "replay.json") return put(request, response, name);
  if (request.method !== "GET" && request.method !== "HEAD") return reply(response, 405);
  const path = join(directory, name);
  let file;
  try {
    file = await open(path, "r");
  } catch (error) {
    if (error.code === "ENOENT") return reply(response, 404);
    throw error;
  }
  response.writeHead(200, {
    "Content-Type": isVideo ? "video/mp4" : "application/json; charset=utf-8",
    "Content-Length": (await file.stat()).size,
  });
  if (request.method === "HEAD") {
    await file.close();
    return response.end();
  }
  file.createReadStream().on("error", () => response.destroy()).pipe(response);
}

createServer((request, response) => {
  handle(request, response).catch((error) => {
    console.error(error.message);
    if (!response.headersSent) reply(response, 500);
    else response.destroy();
  });
}).listen(Number(process.env.PORT || 8080), "0.0.0.0");
