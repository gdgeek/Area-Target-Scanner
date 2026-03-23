#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include "visual_localizer.h"

struct VocabWord {
    int word_id;
    std::vector<unsigned char> descriptor;
    float idf_weight;
};

struct KeyframeData {
    int id;
    cv::Mat pose;              // 4x4 float
    cv::Mat descriptors;       // Nx32 CV_8UC1
    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;
    std::vector<float> global_descriptor; // BoW vector
};

class VisualLocalizer {
public:
    VisualLocalizer();
    ~VisualLocalizer();

    void addVocabularyWord(int word_id, const unsigned char* desc, int desc_len, float idf_weight);
    void addKeyframe(int kf_id, const float* pose_4x4,
                     const unsigned char* descriptors, int desc_count,
                     const float* points3d, const float* points2d);
    bool buildIndex();

    VLDebugInfo getDebugInfo() const { return last_debug_info_; }

    VLResult processFrame(const unsigned char* image_data, int width, int height,
                          float fx, float fy, float cx, float cy,
                          bool has_last_pose, const float* last_pose_4x4);
    void reset();

private:
    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::BFMatcher> matcher_;
    std::vector<VocabWord> vocabulary_;
    std::vector<KeyframeData> keyframes_;
    bool index_built_ = false;

    // Debug diagnostics for last processed frame
    VLDebugInfo last_debug_info_ = {};

    // Algorithm parameters — balanced for real-world AR
    static constexpr int kOrbNFeatures = 1500;
    static constexpr int kMinFeatureCount = 10;
    static constexpr float kLoweRatio = 0.82f;        // 0.78太紧, 0.85太松
    static constexpr int kMinGoodMatches = 10;         // 15太紧, 8太松
    static constexpr int kPnpIterations = 200;
    static constexpr float kPnpReprojError = 10.0f;    // 8太紧, 12太松
    static constexpr double kPnpConfidence = 0.99;
    static constexpr int kMinInlierCount = 10;         // 15太紧, 8太松
    static constexpr float kMaxConfidenceDivisor = 60.0f;
    static constexpr float kNearbyRadius = 10.0f;     // was 5.0: wider search radius
    static constexpr int kMaxNearbyKeyframes = 10;    // was 5: check more candidates
    static constexpr int kGlobalTopK = 20;            // was 10: more BoW candidates
    static constexpr int kAbsoluteDistThreshold = 64;  // Hamming distance fallback threshold (max 256)

    // Internal methods
    std::vector<float> computeBoW(const cv::Mat& descriptors);
    std::vector<KeyframeData*> getNearbyKeyframes(const float* last_pose, float radius, int max_count);
    std::vector<KeyframeData*> getGlobalCandidates(const cv::Mat& descriptors);
    VLResult tryMatchKeyframe(const KeyframeData& kf,
                              const std::vector<cv::KeyPoint>& query_kps,
                              const cv::Mat& query_desc,
                              float fx, float fy, float cx, float cy,
                              int* out_raw_matches = nullptr,
                              int* out_good_matches = nullptr);
    static int hammingDistance(const unsigned char* a, const unsigned char* b, int len);
    static float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
};
