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

    // Algorithm parameters — matching existing C# VisualLocalizationEngine constants
    static constexpr int kOrbNFeatures = 1000;
    static constexpr int kMinFeatureCount = 10;
    static constexpr float kLoweRatio = 0.75f;
    static constexpr int kMinGoodMatches = 20;
    static constexpr int kPnpIterations = 100;
    static constexpr float kPnpReprojError = 8.0f;
    static constexpr double kPnpConfidence = 0.99;
    static constexpr int kMinInlierCount = 20;
    static constexpr float kMaxConfidenceDivisor = 100.0f;
    static constexpr float kNearbyRadius = 5.0f;
    static constexpr int kMaxNearbyKeyframes = 5;
    static constexpr int kGlobalTopK = 10;

    // Internal methods
    std::vector<float> computeBoW(const cv::Mat& descriptors);
    std::vector<KeyframeData*> getNearbyKeyframes(const float* last_pose, float radius, int max_count);
    std::vector<KeyframeData*> getGlobalCandidates(const cv::Mat& descriptors);
    VLResult tryMatchKeyframe(const KeyframeData& kf,
                              const std::vector<cv::KeyPoint>& query_kps,
                              const cv::Mat& query_desc,
                              float fx, float fy, float cx, float cy);
    static int hammingDistance(const unsigned char* a, const unsigned char* b, int len);
    static float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
};
