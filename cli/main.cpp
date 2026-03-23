#include "qwen3_asr.h"
#include "forced_aligner.h"
#include "timing.h"

#include <ggml.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <string>
#include <fstream>

struct cli_params {
    std::string model_path = "models/qwen3-asr-0.6b-f16.gguf";
    std::string aligner_model_path = "";
    std::string audio_path = "";
    std::string output_path = "";
    std::string language = "";
    std::string align_text = "";
    int32_t max_tokens = 1024;
    int32_t n_threads = 4;
    bool print_progress = false;
    bool print_timing = true;
    bool print_tokens = false;
    bool align_mode = false;
    bool transcribe_align_mode = false;
    bool profile = false;
    bool output_srt = false;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <path>     Path to GGUF model (default: models/qwen3-asr-0.6b-f16.gguf)\n");
    fprintf(stderr, "  -f, --audio <path>     Path to audio file (WAV, 16kHz mono) [required]\n");
    fprintf(stderr, "  -o, --output <path>    Output file path (default: stdout)\n");
    fprintf(stderr, "  -l, --language <code>  Language code (optional, e.g. 'korean' for Korean word splitting)\n");
    fprintf(stderr, "  -t, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum tokens to generate (default: 1024)\n");
    fprintf(stderr, "  --progress             Print progress during transcription\n");
    fprintf(stderr, "  --no-timing            Don't print timing information\n");
    fprintf(stderr, "  --tokens               Print token IDs\n");
    fprintf(stderr, "  --profile              Print detailed timing profile (requires QWEN3_ASR_TIMING build)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Forced Alignment:\n");
    fprintf(stderr, "  --align                Enable forced alignment mode\n");
    fprintf(stderr, "  --text <text>          Reference transcript for alignment\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Transcribe + Align:\n");
    fprintf(stderr, "  -a, --transcribe-align Run ASR then forced alignment\n");
    fprintf(stderr, "  --aligner-model <path> Path to forced aligner GGUF model (required with --transcribe-align)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Output Formats:\n");
    fprintf(stderr, "  -osrt, --output-srt    Output result in a SRT file\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  -h, --help             Show this help message\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  Transcription:\n");
    fprintf(stderr, "    %s -m models/qwen3-asr-0.6b-f16.gguf -f sample.wav\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  Forced Alignment:\n");
    fprintf(stderr, "    %s -m models/qwen3-forced-aligner-0.6b-f16.gguf -f sample.wav --align --text \"Hello world\"\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  Transcribe + Align:\n");
    fprintf(stderr, "    %s -m models/qwen3-asr-0.6b-f16.gguf --aligner-model models/qwen3-forced-aligner-0.6b-f16.gguf -f sample.wav --transcribe-align\n", prog);
}

static bool parse_args(int argc, char ** argv, cli_params & params) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        
        if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.model_path = argv[++i];
        } else if (strcmp(arg, "-f") == 0 || strcmp(arg, "--audio") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.audio_path = argv[++i];
        } else if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.output_path = argv[++i];
        } else if (strcmp(arg, "-l") == 0 || strcmp(arg, "--language") == 0 || strcmp(arg, "--lang") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.language = argv[++i];
        } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.n_threads = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--max-tokens") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.max_tokens = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--progress") == 0) {
            params.print_progress = true;
        } else if (strcmp(arg, "--no-timing") == 0) {
            params.print_timing = false;
        } else if (strcmp(arg, "--tokens") == 0) {
            params.print_tokens = true;
        } else if (strcmp(arg, "--profile") == 0) {
            params.profile = true;
        } else if (strcmp(arg, "--align") == 0) {
            params.align_mode = true;
        } else if (strcmp(arg, "-osrt") == 0 || strcmp(arg, "--output-srt") == 0) {
            params.output_srt = true;
        } else if (strcmp(arg, "-a") == 0 || strcmp(arg, "--transcribe-align") == 0) {
            params.transcribe_align_mode = true;
        } else if (strcmp(arg, "--aligner-model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.aligner_model_path = argv[++i];
        } else if (strcmp(arg, "--text") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg);
                return false;
            }
            params.align_text = argv[++i];
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Error: Unknown argument: %s\n", arg);
            return false;
        }
    }
    
    if (params.audio_path.empty()) {
        fprintf(stderr, "Error: Audio file path is required (-f/--audio)\n");
        return false;
    }
    
    if (params.align_mode && params.align_text.empty()) {
        fprintf(stderr, "Error: Reference text is required for alignment mode (--text)\n");
        return false;
    }

    if (params.align_mode && params.transcribe_align_mode) {
        fprintf(stderr, "Error: --align and --transcribe-align cannot be used together\n");
        return false;
    }

    if (params.transcribe_align_mode && params.aligner_model_path.empty()) {
        fprintf(stderr, "Error: --aligner-model is required for --transcribe-align\n");
        return false;
    }
    
    return true;
}

static std::string detect_language(const std::string & asr_language_token) {
    std::string lang = asr_language_token;

    // Trim whitespace
    lang.erase(lang.begin(), std::find_if(lang.begin(), lang.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    lang.erase(std::find_if(lang.rbegin(), lang.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), lang.end());

    // Convert to lowercase
    std::transform(lang.begin(), lang.end(), lang.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Some common mappings if needed
    if (lang == "<|zh|>") return "chinese";
    if (lang == "<|en|>") return "english";
    if (lang == "<|ko|>") return "korean";

    return lang;
}

static std::string extract_transcript(const std::string & asr_text) {
    // Qwen3 ASR text doesn't have "language " prefix in the text itself.
    // It's just the raw transcript.
    return asr_text;
}

static std::string escape_json_string(const std::string & s) {
    std::string result;
    result.reserve(s.size() + 10);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}

static std::string alignment_to_json(const qwen3_asr::alignment_result & result) {
    std::string json = "{\n  \"words\": [\n";

    for (size_t i = 0; i < result.words.size(); ++i) {
        const auto & w = result.words[i];
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "    {\"word\": \"%s\", \"start\": %.3f, \"end\": %.3f}",
                 escape_json_string(w.word).c_str(), w.start, w.end);
        json += buf;
        if (i + 1 < result.words.size()) {
            json += ",";
        }
        json += "\n";
    }

    json += "  ]\n}";
    return json;
}

static std::string to_timestamp(double t_sec, bool comma = false) {
    int64_t msec = static_cast<int64_t>(t_sec * 1000.0);
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);
    return std::string(buf);
}

static std::string alignment_to_srt(const qwen3_asr::alignment_result & result) {
    std::string srt = "";
    if (result.words.empty()) return srt;

    int segment_index = 1;
    size_t i = 0;

    while (i < result.words.size()) {
        std::string text = "";
        double start_time = result.words[i].start;
        double end_time = result.words[i].end;
        int word_count = 0;

        while (i < result.words.size()) {
            const auto & w = result.words[i];

            // Only break on pauses if the pause is very long (e.g. > 3 seconds) 
            // OR if the pause is > 1.5 seconds AND the previous word ended a clause
            bool is_end_of_clause = !text.empty() && 
                (text.back() == ',' || text.back() == '.' || text.back() == '?' || text.back() == '!' ||
                 text.find("\xEF\xBC\x8C") != std::string::npos || // ，
                 text.find("\xE3\x80\x82") != std::string::npos);  // 。

            if (word_count > 0 && (w.start - end_time > 3.0 || (w.start - end_time > 1.5 && is_end_of_clause))) {
                break;
            }

            // For non-Asian languages, we usually need spaces between words
            if (word_count > 0 && !text.empty() && 
                static_cast<unsigned char>(text.back()) < 0x80 &&
                static_cast<unsigned char>(w.word.front()) < 0x80 &&
                w.word.find_first_of(".,!?") != 0) {
                text += " ";
            }

            text += w.word;
            // Update end time only if the word has a valid end time
            if (w.end > end_time) {
                end_time = w.end;
            }
            word_count++;
            i++;

            // Break on terminal punctuation (sentences)
            if (w.word.find("\xE3\x80\x82") != std::string::npos || // 。
                w.word.find("\xEF\xBC\x9F") != std::string::npos || // ？
                w.word.find("\xEF\xBC\x81") != std::string::npos || // ！
                w.word.find(".") != std::string::npos ||
                w.word.find("?") != std::string::npos ||
                w.word.find("!") != std::string::npos ||
                w.word.find("\n") != std::string::npos) {
                break;
            }

            // Break if sentence gets too long (prevent endless blocks)
            if (word_count >= 20 || text.size() >= 80) {
                break;
            }
        }

        char buf[64];
        snprintf(buf, sizeof(buf), "%d\n", segment_index++);
        srt += buf;

        srt += to_timestamp(start_time, true) + " --> " + to_timestamp(end_time, true) + "\n";
        srt += text + "\n\n";
    }

    return srt;
}
static std::string find_korean_dict(const std::string & model_path) {
    auto dir_of = [](const std::string & path) -> std::string {
        size_t pos = path.find_last_of("/\\");
        return (pos != std::string::npos) ? path.substr(0, pos) : ".";
    };

    std::vector<std::string> candidates = {
        dir_of(model_path) + "/../assets/korean_dict_jieba.dict",
        dir_of(model_path) + "/assets/korean_dict_jieba.dict",
        "assets/korean_dict_jieba.dict",
    };

    for (const auto & p : candidates) {
        std::ifstream f(p);
        if (f.good()) return p;
    }
    return "";
}

static int run_alignment(const cli_params & params) {
    fprintf(stderr, "qwen3-asr-cli (Forced Alignment Mode)\n");
    fprintf(stderr, "  Model: %s\n", params.model_path.c_str());
    fprintf(stderr, "  Audio: %s\n", params.audio_path.c_str());
    fprintf(stderr, "  Text: %s\n", params.align_text.c_str());
    if (!params.language.empty()) {
        fprintf(stderr, "  Language: %s\n", params.language.c_str());
    }
    fprintf(stderr, "\n");
    
    qwen3_asr::ForcedAligner aligner;
    
    if (!aligner.load_model(params.model_path)) {
        fprintf(stderr, "Error: %s\n", aligner.get_error().c_str());
        return 1;
    }
    
    if (params.language == "korean") {
        std::string dict_path = find_korean_dict(params.model_path);
        if (dict_path.empty()) {
            fprintf(stderr, "Warning: Korean dictionary not found. Falling back to whitespace splitting.\n");
        } else {
            if (!aligner.load_korean_dict(dict_path)) {
                fprintf(stderr, "Warning: Failed to load Korean dictionary from %s\n", dict_path.c_str());
            }
        }
    }
    
    fprintf(stderr, "Model loaded. Running alignment...\n");
    
    auto result = aligner.align(params.audio_path, params.align_text, params.language);
    
    if (!result.success) {
        fprintf(stderr, "Error: %s\n", result.error_msg.c_str());
        return 1;
    }
    
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Mel spectrogram: %lld ms\n", (long long)result.t_mel_ms);
        fprintf(stderr, "  Audio encoding:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Text decoding:   %lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
        fprintf(stderr, "  Words aligned:   %zu\n", result.words.size());
    }
    
    std::string string_output = params.output_srt ? alignment_to_srt(result) : alignment_to_json(result);
    
    if (params.output_path.empty()) {
        printf("%s\n", string_output.c_str());
    } else {
        std::ofstream out(params.output_path);
        if (!out) {
            fprintf(stderr, "Error: Failed to open output file: %s\n", params.output_path.c_str());
            return 1;
        }
        out << string_output << "\n";
        fprintf(stderr, "Output written to: %s\n", params.output_path.c_str());
    }
    
    if (params.profile) {
        QWEN3_TIMER_REPORT();
    }
    
    return 0;
}

static int run_transcription(const cli_params & params) {
    fprintf(stderr, "qwen3-asr-cli\n");
    fprintf(stderr, "  Model: %s\n", params.model_path.c_str());
    fprintf(stderr, "  Audio: %s\n", params.audio_path.c_str());
    fprintf(stderr, "  Threads: %d\n", params.n_threads);
    fprintf(stderr, "\n");
    
    qwen3_asr::Qwen3ASR asr;
    
    if (!asr.load_model(params.model_path)) {
        fprintf(stderr, "Error: %s\n", asr.get_error().c_str());
        return 1;
    }
    
    qwen3_asr::transcribe_params tp;
    tp.max_tokens = params.max_tokens;
    tp.language = params.language;
    tp.n_threads = params.n_threads;
    tp.print_progress = params.print_progress;
    tp.print_timing = params.print_timing;
    
    auto result = asr.transcribe(params.audio_path, tp);
    
    if (!result.success) {
        fprintf(stderr, "Error: %s\n", result.error_msg.c_str());
        return 1;
    }
    
    if (params.print_tokens) {
        fprintf(stderr, "\nTokens (%zu):\n", result.tokens.size());
        for (size_t i = 0; i < result.tokens.size(); ++i) {
            fprintf(stderr, "  [%zu] %d\n", i, result.tokens[i]);
        }
        fprintf(stderr, "\n");
    }
    
    if (params.output_path.empty()) {
        printf("%s\n", result.text.c_str());
    } else {
        std::ofstream out(params.output_path);
        if (!out) {
            fprintf(stderr, "Error: Failed to open output file: %s\n", params.output_path.c_str());
            return 1;
        }
        out << result.text << "\n";
        fprintf(stderr, "Output written to: %s\n", params.output_path.c_str());
    }
    
    if (params.profile) {
        QWEN3_TIMER_REPORT();
    }
    
    return 0;
}

static int run_transcribe_and_align(const cli_params & params) {
    fprintf(stderr, "qwen3-asr-cli (Transcribe + Align Mode)\n");
    fprintf(stderr, "  ASR Model: %s\n", params.model_path.c_str());
    fprintf(stderr, "  Aligner Model: %s\n", params.aligner_model_path.c_str());
    fprintf(stderr, "  Audio: %s\n", params.audio_path.c_str());
    fprintf(stderr, "  Threads: %d\n", params.n_threads);
    fprintf(stderr, "\n");

    qwen3_asr::Qwen3ASR asr;
    if (!asr.load_model(params.model_path)) {
        fprintf(stderr, "Error (ASR): %s\n", asr.get_error().c_str());
        return 1;
    }

    qwen3_asr::ForcedAligner aligner;
    if (!aligner.load_model(params.aligner_model_path)) {
        fprintf(stderr, "Error (Aligner): %s\n", aligner.get_error().c_str());
        return 1;
    }

    std::vector<float> all_samples;
    int sample_rate;
    if (!load_wav(params.audio_path, all_samples, sample_rate)) {
        fprintf(stderr, "Error: Failed to load audio file: %s\n", params.audio_path.c_str());
        return 1;
    }
    if (sample_rate != 16000) {
        fprintf(stderr, "Error: Audio must be 16kHz\n");
        return 1;
    }

    const int chunk_size_samples = 30 * 16000;
    const int chunk_stride_samples = 28 * 16000; // 2 seconds overlap
    int n_chunks = (all_samples.size() > chunk_size_samples) 
                   ? (all_samples.size() - chunk_size_samples + chunk_stride_samples - 1) / chunk_stride_samples + 1 
                   : 1;

    qwen3_asr::alignment_result full_align_result;
    full_align_result.success = true;
    full_align_result.t_total_ms = 0;
    int64_t total_asr_ms = 0;
    int64_t total_align_ms = 0;

    std::string global_lang = params.language;

    for (int c = 0; c < n_chunks; ++c) {
        int start = c * chunk_stride_samples;
        int end = std::min((int)all_samples.size(), start + chunk_size_samples);
        int chunk_len = end - start;
        
        fprintf(stderr, "\n--- Processing Chunk %d/%d (%.2fs - %.2fs) ---\n", 
                c + 1, n_chunks, start / 16000.0f, end / 16000.0f);

        qwen3_asr::transcribe_params tp;
        tp.max_tokens = params.max_tokens;
        tp.language = global_lang;
        tp.n_threads = params.n_threads;
        tp.print_progress = params.print_progress;
        tp.print_timing = params.print_timing;

        auto asr_result = asr.transcribe(all_samples.data() + start, chunk_len, tp);
        if (!asr_result.success) {
            fprintf(stderr, "Error (ASR) in chunk %d: %s\n", c + 1, asr_result.error_msg.c_str());
            continue;
        }

        std::string detected_lang = detect_language(asr_result.language);
        if (global_lang.empty() && !detected_lang.empty()) {
            global_lang = detected_lang;
            if (global_lang == "korean") {
                std::string dict_path = find_korean_dict(params.aligner_model_path);
                if (!dict_path.empty()) aligner.load_korean_dict(dict_path);
            }
        }
        
        std::string align_lang = global_lang.empty() ? detected_lang : global_lang;
        std::string transcript = extract_transcript(asr_result.text);
        
        fprintf(stderr, "  Detected language: %s\n", detected_lang.empty() ? "(none)" : detected_lang.c_str());
        fprintf(stderr, "  Alignment language: %s\n", align_lang.empty() ? "(none)" : align_lang.c_str());
        fprintf(stderr, "  Transcript: %s\n", transcript.c_str());
        
        if (transcript.empty()) continue;

        auto align_result = aligner.align(all_samples.data() + start, chunk_len, transcript, align_lang);
        if (!align_result.success) {
            fprintf(stderr, "Error (Aligner) in chunk %d: %s\n", c + 1, align_result.error_msg.c_str());
            continue;
        }

        float time_offset = start / 16000.0f;
        
        // Define strict ownership boundaries for this chunk based on the overlapping stride
        double chunk_own_start = time_offset;
        double chunk_own_end = (c == n_chunks - 1) ? 999999.0 : (time_offset + (chunk_stride_samples / 16000.0f));
        
        for (auto & w : align_result.words) {
            double abs_start = w.start + time_offset;
            double abs_end = w.end + time_offset;
            
            // Hard partitioning: This chunk only "owns" words that start within its active stride window.
            // This prevents trailing truncated words (e.g. "For your.") in Chunk N from duplicating with Chunk N+1.
            if (abs_start >= chunk_own_start && abs_start < chunk_own_end) {
                w.start = abs_start;
                w.end = abs_end;
                full_align_result.words.push_back(w);
            }
        }
        
        total_asr_ms += asr_result.t_total_ms;
        total_align_ms += align_result.t_total_ms;
        full_align_result.t_total_ms += (asr_result.t_total_ms + align_result.t_total_ms);
    }

    if (params.print_timing) {
        fprintf(stderr, "\nCombined Timing:\n");
        fprintf(stderr, "  ASR:           %lld ms\n", (long long) total_asr_ms);
        fprintf(stderr, "  Alignment:     %lld ms\n", (long long) total_align_ms);
        fprintf(stderr, "  Total:         %lld ms\n", (long long) full_align_result.t_total_ms);
        fprintf(stderr, "  Words aligned: %zu\n", full_align_result.words.size());
    }

    std::string string_output = params.output_srt ? alignment_to_srt(full_align_result) : alignment_to_json(full_align_result);

    if (params.output_path.empty()) {
        printf("%s\n", string_output.c_str());
    } else {
        std::ofstream out(params.output_path);
        if (!out) {
            fprintf(stderr, "Error: Failed to open output file: %s\n", params.output_path.c_str());
            return 1;
        }
        out << string_output << "\n";
        fprintf(stderr, "Output written to: %s\n", params.output_path.c_str());
    }

    if (params.profile) {
        QWEN3_TIMER_REPORT();
    }

    return 0;
}

static void ggml_log_callback_quiet(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    ggml_log_set(ggml_log_callback_quiet, nullptr);

    cli_params params;
    
    if (!parse_args(argc, argv, params)) {
        fprintf(stderr, "\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (params.transcribe_align_mode) {
        return run_transcribe_and_align(params);
    }

    if (params.align_mode) {
        return run_alignment(params);
    } else {
        return run_transcription(params);
    }
}
