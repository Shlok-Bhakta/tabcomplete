import { describe, test, expect } from "bun:test";
import { parseArgs } from "../lib.ts";

const ALLOWED = new Set(["db", "file", "repo"]);

describe("parseArgs", () => {
  test("key value form", () => {
    expect(parseArgs(["--db", "x", "--file", "y"], ALLOWED)).toEqual({ db: "x", file: "y" });
  });
  test("key=value form", () => {
    expect(parseArgs(["--db=x", "--file=y"], ALLOWED)).toEqual({ db: "x", file: "y" });
  });
  test("mixed forms", () => {
    expect(parseArgs(["--db=x", "--file", "y"], ALLOWED)).toEqual({ db: "x", file: "y" });
  });
  test("unknown flag throws", () => {
    expect(() => parseArgs(["--nope", "1"], ALLOWED)).toThrow(/unknown flag/);
    expect(() => parseArgs(["--nope=1"], ALLOWED)).toThrow(/unknown flag/);
  });
  test("positional throws", () => {
    expect(() => parseArgs(["somefile"], ALLOWED)).toThrow(/positional/);
    expect(() => parseArgs(["--db", "x", "extra"], ALLOWED)).toThrow(/positional/);
  });
});
