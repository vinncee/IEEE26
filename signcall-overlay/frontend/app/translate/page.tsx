"use client";

import React, { useEffect, useRef, useState } from "react";
import { Navbar } from "@/components/navbar";
import { useSignBridgeWS } from "@/hooks/use-signbridge-ws";
import { cn } from "@/lib/utils";

export default function TranslatePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraPermission, setCameraPermission] = useState<"granted" | "denied" | "prompt" | null>(null);
  const [isLoadingCamera, setIsLoadingCamera] = useState(false);
  const [captionSize, setCaptionSize] = useState(24);

  const { connection, latestCaption, connect, disconnect } = useSignBridgeWS();

  // Start camera stream
  const getCamera = async () => {
    setIsLoadingCamera(true);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setStream(mediaStream);
      setError(null);
      setCameraPermission("granted");
      setIsLoadingCamera(false);
      return mediaStream;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to access camera";
      setError(errorMsg);
      setCameraPermission("denied");
      setIsLoadingCamera(false);
      console.error("Camera error:", err);
      return null;
    }
  };

  // Cleanup stream on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream]);

  // Connect to WebSocket
  const handleStartRecording = async () => {
    let activeStream = stream;
    
    // If no stream, request camera first
    if (!activeStream) {
      activeStream = await getCamera();
      if (!activeStream) {
        return; // Camera request failed
      }
    }
    
    setIsRecording(true);
    connect();
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    disconnect();
    // Stop camera stream
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    setStream(null);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "bg-emerald-100 text-emerald-900";
    if (confidence >= 0.55) return "bg-yellow-100 text-yellow-900";
    return "bg-red-100 text-red-900";
  };

  const getModeLabel = (mode: string) => {
    const labels: Record<string, string> = {
      template: "Template",
      llm: "LLM",
      uncertain: "Uncertain",
    };
    return labels[mode] || mode;
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="flex-1 px-6 py-8">
        <div className="mx-auto max-w-6xl">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-foreground">
              Sign Language Translator
            </h1>
            <p className="mt-2 text-lg text-muted-foreground">
              Position yourself in good lighting and sign naturally. Real-time
              Position yourself in good lighting and sign naturally. Real-time
              captions will appear below.
            </p>
          </div>

          {/* Camera Feed */}
          <div className="mb-6 rounded-2xl border-2 border-border overflow-hidden bg-black">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="h-auto w-full bg-black"
              style={{ transform: "scaleX(-1)", minHeight: "400px" }}
            />
          </div>

          {/* Caption Display */}
          <div className="mb-6 rounded-2xl border-2 border-border bg-card p-6">
            <h3 className="mb-4 text-sm font-semibold text-muted-foreground uppercase tracking-wide">
              Live Captions
            </h3>
            {isRecording && latestCaption ? (
              <div>
                <div className="mb-4 rounded-lg bg-secondary p-4 min-h-20">
                  <p className="font-semibold text-foreground" style={{ fontSize: `${captionSize}px` }}>
                    {latestCaption.caption || "Waiting for input..."}
                  </p>
                </div>
                <div className="flex gap-2 flex-wrap">
                  <div className={cn("rounded-lg px-3 py-1 text-sm font-medium", getConfidenceColor(latestCaption.confidence))}>
                    Confidence: {(latestCaption.confidence * 100).toFixed(0)}%
                  </div>
                  <div className="rounded-lg bg-primary/20 px-3 py-1 text-sm font-medium text-primary">
                    Mode: {getModeLabel(latestCaption.mode)}
                  </div>
                </div>
              </div>
            ) : isRecording ? (
              <div className="flex items-center justify-center text-center text-muted-foreground py-8 min-h-20">
                <p>Waiting for sign language input...</p>
              </div>
            ) : (
              <div className="flex items-center justify-center text-center text-muted-foreground py-8 min-h-20">
                <p>Start translating to see captions</p>
              </div>
            )}
          </div>

          {/* Caption Size Slider */}
          {isRecording && (
            <div className="mb-6 rounded-lg bg-card p-4 border border-border">
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-semibold text-muted-foreground">Caption Size</label>
                <span className="text-sm font-medium text-foreground">{captionSize}px</span>
              </div>
              <input
                type="range"
                min="14"
                max="48"
                value={captionSize}
                onChange={(e) => setCaptionSize(Number(e.target.value))}
                className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
              />
            </div>
          )}

          {/* Error Display */}
          {error && cameraPermission === "denied" && (
            <div className="mb-6 rounded-lg border border-destructive/50 bg-destructive/10 p-4">
              <p className="text-destructive font-medium">Camera Access Denied</p>
              <p className="mt-1 text-sm text-destructive/80">
                {error}. Please check your browser permissions.
              </p>
            </div>
          )}

          {/* Connection Status */}
          <div className="mb-6 flex items-center gap-2">
            <div
              className={cn(
                "h-3 w-3 rounded-full",
                connection.status === "connected"
                  ? "bg-emerald-500 animate-pulse"
                  : connection.status === "connecting"
                    ? "bg-yellow-500 animate-pulse"
                    : connection.status === "error"
                      ? "bg-red-500"
                      : "bg-gray-400"
              )}
            />
            <span className="text-sm font-medium text-muted-foreground">
              {connection.status === "connected"
                ? "Connected to backend"
                : connection.status === "connecting"
                  ? "Connecting..."
                  : connection.status === "error"
                    ? `Error: ${connection.message}`
                    : "Disconnected"}
            </span>
          </div>

          {/* Controls */}
          <div className="flex gap-4">
            <button
              onClick={handleStartRecording}
              disabled={isRecording || isLoadingCamera}
              className="rounded-xl bg-primary px-8 py-4 font-semibold text-primary-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:shadow-lg"
            >
              {isLoadingCamera
                ? "Requesting camera..."
                : isRecording
                  ? "Recording..."
                  : "Start Translation"}
            </button>
            <button
              onClick={handleStopRecording}
              disabled={!isRecording}
              className="rounded-xl border-2 border-border bg-transparent px-8 py-4 font-semibold text-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:bg-secondary"
            >
              Stop
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
