"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface CaptionMessage {
  type: "caption";
  caption: string;
  mode: "template" | "llm" | "uncertain";
  confidence: number;
  timestamp: number;
  session?: string;
  user?: string;
}

export interface ConnectionState {
  status: "disconnected" | "connecting" | "connected" | "error";
  message?: string;
}

const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/video";

export function useSignBridgeWS() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connection, setConnection] = useState<ConnectionState>({
    status: "disconnected",
  });
  const [captions, setCaptions] = useState<CaptionMessage[]>([]);
  const [latestCaption, setLatestCaption] = useState<CaptionMessage | null>(
    null,
  );

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setConnection({ status: "connecting" });

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnection({ status: "connected" });
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "caption") {
            const caption: CaptionMessage = {
              type: "caption",
              caption: data.caption || "",
              mode: data.mode || "uncertain",
              confidence: data.confidence || 0,
              timestamp: data.ts || Date.now(),
              session: data.session,
              user: data.user,
            };
            setLatestCaption(caption);
            setCaptions((prev) => [...prev.slice(-49), caption]);
          }
        } catch {
          // Non-JSON message, ignore
        }
      };

      ws.onerror = () => {
        setConnection({ status: "error", message: "Connection failed" });
      };

      ws.onclose = () => {
        setConnection({ status: "disconnected" });
        wsRef.current = null;
      };
    } catch {
      setConnection({
        status: "error",
        message: "Could not connect to server",
      });
    }
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnection({ status: "disconnected" });
  }, []);

  const sendFrame = useCallback((frameData: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "frame",
          data: frameData,
          ts: Date.now(),
        }),
      );
    }
  }, []);

  const sendCorrection = useCallback(
    (incorrectToken: string, correctToken: string) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: "correction",
            incorrect_token: incorrectToken,
            correct_token: correctToken,
            ts: Date.now(),
          }),
        );
      }
    },
    [],
  );

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return {
    connection,
    captions,
    latestCaption,
    connect,
    disconnect,
    sendFrame,
    sendCorrection,
  };
}
