export type FrameIn = {
  type: "frame";
  session: string;
  user: string;
  ts: number;
  image_jpeg_b64: string;
  style?: "concise" | "detailed";
};

export type CaptionOut = {
  type: "caption";
  session: string;
  user: string;
  ts: number;
  caption: string;
  confidence: number;
  mode: "template" | "llm" | "uncertain";
};

export type CorrectionIn = {
  type: "correction";
  session: string;
  user: string;
  ts: number;
  incorrect_token: string;
  correct_token: string;
};

export type WsIn = CaptionOut;
export type WsOut = FrameIn | CorrectionIn;
