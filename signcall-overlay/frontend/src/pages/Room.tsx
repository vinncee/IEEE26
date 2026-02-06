import { useEffect, useRef, useState } from "react";
import VideoTile from "../components/VideoTile";
import Controls from "../components/Controls";
import { createWs } from "../services/ws";
import { captureJpegBase64 } from "../services/frameCapture";
import { getLocalStream } from "../services/webrtc";
import type { CaptionOut, CorrectionIn, FrameIn } from "../types/messages";

export default function Room() {
  const session = "room1";
  const user = "signerA";
  const wsUrl = "ws://localhost:8000/ws";

  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const [connected, setConnected] = useState(false);
  const [caption, setCaption] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [mode, setMode] = useState<CaptionOut["mode"]>("template");

  const [modePref, setModePref] = useState<"concise" | "detailed">("concise");
  const modePrefRef = useRef(modePref);
  useEffect(() => { modePrefRef.current = modePref; }, [modePref]);

  const wsRef = useRef<ReturnType<typeof createWs> | null>(null);

  useEffect(() => {
    if (wsRef.current) return;
    wsRef.current = createWs(wsUrl, {
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
      onMessage: (msg) => {
        if (msg.type === "caption") {
          setCaption(msg.caption);
          setConfidence(msg.confidence);
          setMode(msg.mode);
        }
      }
    });

    return () => {
      wsRef.current?.ws.close();
      wsRef.current = null;
    };
  }, []);

  useEffect(() => {
    getLocalStream().then(setStream).catch(console.error);
  }, []);

  // frame loop
  useEffect(() => {
    let alive = true;
    const fps = 8;
    const intervalMs = Math.floor(1000 / fps);

    const tick = async () => {
      if (!alive) return;
      const v = videoRef.current;
      if (v && v.readyState >= 2 && wsRef.current) {
        const b64 = await captureJpegBase64(v);
        const msg: FrameIn = {
          type: "frame",
          session,
          user,
          ts: Date.now(),
          image_jpeg_b64: b64,
          style: modePrefRef.current
        };
        wsRef.current.send(msg);
      }
      setTimeout(tick, intervalMs);
    };

    tick();
    return () => {
      alive = false;
    };
  }, []);

  const sendCorrection = () => {
    const msg: CorrectionIn = {
      type: "correction",
      session,
      user,
      ts: Date.now(),
      incorrect_token: "SLOW",
      correct_token: "REPEAT"
    };
    wsRef.current?.send(msg);
  };

  return (
    <>
      <Controls connected={connected} modePref={modePref} setModePref={setModePref} sendCorrection={sendCorrection} />
      <div className="row">
        <VideoTile
          title="Local (Signer)"
          stream={stream}
          caption={caption}
          confidence={confidence}
          mode={mode}
          videoRef={videoRef}
        />
      </div>
      <p style={{ marginTop: 12, color: "#555" }}>
        This is a minimal single-stream demo: backend returns captions for frames sent over WebSocket.
        Replace with full WebRTC room later.
      </p>
    </>
  );
}
