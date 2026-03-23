#include "visual_localizer_impl.h"
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <climits>
#include <numeric>

// ---------------------------------------------------------------------------
// Helper: create a LOST VLResult with identity pose
// ---------------------------------------------------------------------------
static VLResult makeLostResult() {
    VLResult r;
    r.state = 2; // LOST
    r.confidence = 0.0f;
    r.matched_features = 0;
    std::fill(std::begin(r.pose), std::end(r.pose), 0.0f);
    r.pose[0] = r.pose[5] = r.pose[10] = r.pose[15] = 1.0f; // identity
    return r;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
VisualLocalizer::VisualLocalizer()
    : orb_(cv::ORB::create(kOrbNFeatures))
    , matcher_(cv::BFMatcher::create(cv::NORM_HAMMING))
    , index_built_(false)
{
}

VisualLocalizer::~VisualLocalizer() = default;

// ---------------------------------------------------------------------------
// addVocabularyWord — deep-copy a single vocabulary word
// ---------------------------------------------------------------------------
void VisualLocalizer::addVocabularyWord(int word_id, const unsigned char* desc,
                                         int desc_len, float idf_weight) {
    VocabWord w;
    w.word_id = word_id;
    w.descriptor.assign(desc, desc + desc_len);
    w.idf_weight = idf_weight;
    vocabulary_.push_back(std::move(w));
}

// ---------------------------------------------------------------------------
// addKeyframe — deep-copy pose, descriptors, 3D/2D points; compute BoW
// ---------------------------------------------------------------------------
void VisualLocalizer::addKeyframe(int kf_id, const float* pose_4x4,
                                   const unsigned char* descriptors,
                                   int desc_count,
                                   const float* points3d,
                                   const float* points2d) {
    KeyframeData kf;
    kf.id = kf_id;

    // Deep-copy 4x4 pose (row-major float)
    kf.pose = cv::Mat(4, 4, CV_32F);
    std::memcpy(kf.pose.data, pose_4x4, 16 * sizeof(float));

    // Deep-copy ORB descriptors (each 32 bytes)
    kf.descriptors = cv::Mat(desc_count, 32, CV_8UC1);
    std::memcpy(kf.descriptors.data, descriptors,
                static_cast<size_t>(desc_count) * 32);

    // Deep-copy 3D points (x,y,z per point)
    kf.points3d.resize(desc_count);
    for (int i = 0; i < desc_count; i++) {
        kf.points3d[i] = cv::Point3f(points3d[i * 3],
                                      points3d[i * 3 + 1],
                                      points3d[i * 3 + 2]);
    }

    // Deep-copy 2D points (x,y per point)
    kf.points2d.resize(desc_count);
    for (int i = 0; i < desc_count; i++) {
        kf.points2d[i] = cv::Point2f(points2d[i * 2],
                                      points2d[i * 2 + 1]);
    }

    // Compute BoW global descriptor for this keyframe
    kf.global_descriptor = computeBoW(kf.descriptors);

    keyframes_.push_back(std::move(kf));
}

// ---------------------------------------------------------------------------
// buildIndex — mark data loading complete
// ---------------------------------------------------------------------------
bool VisualLocalizer::buildIndex() {
    index_built_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// reset — clear tracking state
// ---------------------------------------------------------------------------
void VisualLocalizer::reset() {
    // No persistent tracking state in the native layer beyond the database.
    // The C# side manages LastValidPose. This is provided for future use
    // and to match the C API contract.
}

// ---------------------------------------------------------------------------
// processFrame — full localization pipeline for a single grayscale frame
// ---------------------------------------------------------------------------
VLResult VisualLocalizer::processFrame(const unsigned char* image_data,
                                        int width, int height,
                                        float fx, float fy, float cx, float cy,
                                        bool has_last_pose,
                                        const float* last_pose_4x4) {
    VLResult lost = makeLostResult();

    // Reset debug info
    last_debug_info_ = {};
    last_debug_info_.best_kf_id = -1;

    if (!image_data || width <= 0 || height <= 0)
        return lost;

    // Step 1: Wrap image data as cv::Mat (zero-copy)
    cv::Mat gray(height, width, CV_8UC1,
                 const_cast<unsigned char*>(image_data));

    // Step 2: ORB detect + compute
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

    last_debug_info_.orb_keypoints = static_cast<int>(keypoints.size());

    if (static_cast<int>(keypoints.size()) < kMinFeatureCount || descriptors.empty())
        return lost;

    // Step 3: Candidate keyframe selection
    std::vector<KeyframeData*> candidates;
    if (has_last_pose && last_pose_4x4) {
        candidates = getNearbyKeyframes(last_pose_4x4, kNearbyRadius,
                                         kMaxNearbyKeyframes);
        if (candidates.empty())
            candidates = getGlobalCandidates(descriptors);
    } else {
        candidates = getGlobalCandidates(descriptors);
    }

    last_debug_info_.candidate_keyframes = static_cast<int>(candidates.size());

    if (candidates.empty())
        return lost;

    // Step 4: Try matching each candidate, keep best by inlier count
    VLResult best = lost;
    int best_inliers = 0;
    int best_raw = 0, best_good = 0;

    for (auto* kf : candidates) {
        int raw_matches = 0, good_matches_count = 0;
        VLResult result = tryMatchKeyframe(*kf, keypoints, descriptors,
                                            fx, fy, cx, cy,
                                            &raw_matches, &good_matches_count);
        if (raw_matches > best_raw) best_raw = raw_matches;
        if (good_matches_count > best_good) {
            best_good = good_matches_count;
        }
        if (result.state == 1 && result.matched_features > best_inliers) {
            best = result;
            best_inliers = result.matched_features;
            last_debug_info_.best_kf_id = kf->id;
        }
    }

    last_debug_info_.best_raw_matches = best_raw;
    last_debug_info_.best_good_matches = best_good;
    last_debug_info_.best_inliers = best_inliers;

    return best;
}

// ---------------------------------------------------------------------------
// tryMatchKeyframe — BFMatcher KNN → Lowe ratio → 3D-2D → PnP RANSAC → pose
// ---------------------------------------------------------------------------
VLResult VisualLocalizer::tryMatchKeyframe(const KeyframeData& kf,
                                            const std::vector<cv::KeyPoint>& query_kps,
                                            const cv::Mat& query_desc,
                                            float fx, float fy,
                                            float cx, float cy,
                                            int* out_raw_matches,
                                            int* out_good_matches) {
    VLResult lost = makeLostResult();
    if (out_raw_matches) *out_raw_matches = 0;
    if (out_good_matches) *out_good_matches = 0;

    if (kf.descriptors.empty())
        return lost;

    // KNN match (k=2)
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(query_desc, kf.descriptors, knn_matches, 2);

    if (out_raw_matches) *out_raw_matches = static_cast<int>(knn_matches.size());

    // Lowe ratio test
    std::vector<cv::DMatch> good_matches;
    for (const auto& m : knn_matches) {
        if (m.size() >= 2 &&
            m[0].distance < kLoweRatio * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    // Fallback: if Lowe ratio yields too few matches, use absolute distance threshold
    if (static_cast<int>(good_matches.size()) < kMinGoodMatches) {
        good_matches.clear();
        for (const auto& m : knn_matches) {
            if (!m.empty() && m[0].distance < kAbsoluteDistThreshold) {
                good_matches.push_back(m[0]);
            }
        }
    }

    if (static_cast<int>(good_matches.size()) < kMinGoodMatches)
        return lost;

    if (out_good_matches) *out_good_matches = static_cast<int>(good_matches.size());

    // Build 3D-2D correspondences from good matches
    std::vector<cv::Point3f> obj_pts;
    std::vector<cv::Point2f> img_pts;
    for (const auto& match : good_matches) {
        int kf_idx = match.trainIdx;
        int q_idx = match.queryIdx;
        if (kf_idx < static_cast<int>(kf.points3d.size()) &&
            q_idx < static_cast<int>(query_kps.size())) {
            obj_pts.push_back(kf.points3d[kf_idx]);
            img_pts.push_back(query_kps[q_idx].pt);
        }
    }

    if (static_cast<int>(obj_pts.size()) < kMinGoodMatches)
        return lost;

    // Camera intrinsic matrix
    cv::Mat camera_mat = (cv::Mat_<double>(3, 3) <<
        static_cast<double>(fx), 0.0, static_cast<double>(cx),
        0.0, static_cast<double>(fy), static_cast<double>(cy),
        0.0, 0.0, 1.0);

    // PnP RANSAC
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(obj_pts, img_pts, camera_mat, cv::noArray(),
                        rvec, tvec, false,
                        kPnpIterations, kPnpReprojError,
                        kPnpConfidence, inliers);

    int inlier_count = inliers.rows;
    if (inlier_count < kMinInlierCount)
        return lost;

    // Compose pose matrix from rvec/tvec via Rodrigues
    // solvePnPRansac returns world-to-camera extrinsic [R|t] in OpenCV
    // camera convention (x-right, y-down, z-forward).
    // Unity/ARKit uses (x-right, y-up, z-back).
    //
    // The 3D points fed to PnP are already in ARKit world coordinates (Y-up),
    // so PnP output [R|t] maps ARKit-world → OpenCV-camera.
    // The relationship is: R_opencv = flip * R_arkit_w2c, where flip = diag(1,-1,-1).
    // Therefore: R_arkit_w2c = flip * R_opencv  (left-multiply only).
    // Translation: t_arkit = flip * t_opencv.
    //
    // IMPORTANT: Do NOT right-multiply by flip — the world coordinates are
    // already in ARKit convention and must not be flipped.
    cv::Mat rot_mat;
    cv::Rodrigues(rvec, rot_mat);

    VLResult result;
    result.state = 1; // TRACKING
    result.confidence = std::min(1.0f,
        static_cast<float>(inlier_count) / kMaxConfidenceDivisor);
    result.matched_features = inlier_count;

    // Fill row-major 4x4 pose: R' = flip * R_opencv, t' = flip * t_opencv
    float R_raw[3][3];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R_raw[r][c] = static_cast<float>(rot_mat.at<double>(r, c));

    float t_raw[3];
    for (int r = 0; r < 3; r++)
        t_raw[r] = static_cast<float>(tvec.at<double>(r, 0));

    // Apply flip = diag(1,-1,-1) on the LEFT only: R' = flip * R, t' = flip * t
    float flip[3] = {1.0f, -1.0f, -1.0f};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++)
            result.pose[r * 4 + c] = flip[r] * R_raw[r][c];
        result.pose[r * 4 + 3] = flip[r] * t_raw[r];
    }
    result.pose[12] = 0.0f;
    result.pose[13] = 0.0f;
    result.pose[14] = 0.0f;
    result.pose[15] = 1.0f;

    return result;
}

// ---------------------------------------------------------------------------
// computeBoW — for each descriptor, find nearest vocab word, accumulate IDF,
//              then L2-normalize the resulting vector
// ---------------------------------------------------------------------------
std::vector<float> VisualLocalizer::computeBoW(const cv::Mat& descriptors) {
    int vocab_size = static_cast<int>(vocabulary_.size());
    if (vocab_size == 0)
        return {};

    std::vector<float> bow(vocab_size, 0.0f);

    for (int i = 0; i < descriptors.rows; i++) {
        const unsigned char* desc = descriptors.ptr<unsigned char>(i);
        int best_word = 0;
        int best_dist = INT_MAX;

        for (int w = 0; w < vocab_size; w++) {
            int len = std::min(32,
                static_cast<int>(vocabulary_[w].descriptor.size()));
            int dist = hammingDistance(desc, vocabulary_[w].descriptor.data(),
                                       len);
            if (dist < best_dist) {
                best_dist = dist;
                best_word = w;
            }
        }
        bow[best_word] += vocabulary_[best_word].idf_weight;
    }

    // L2 normalize
    float norm = 0.0f;
    for (float v : bow)
        norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-9f) {
        for (float& v : bow)
            v /= norm;
    }

    return bow;
}

// ---------------------------------------------------------------------------
// getNearbyKeyframes — filter keyframes within Euclidean radius of last pose,
//                      sorted by distance, return top max_count
// ---------------------------------------------------------------------------
std::vector<KeyframeData*> VisualLocalizer::getNearbyKeyframes(
    const float* last_pose, float radius, int max_count) {

    // Extract translation from 4x4 row-major pose: elements [3], [7], [11]
    float lx = last_pose[3];
    float ly = last_pose[7];
    float lz = last_pose[11];

    struct Candidate {
        KeyframeData* kf;
        float dist;
    };
    std::vector<Candidate> within_radius;

    for (auto& kf : keyframes_) {
        // Extract keyframe translation from its 4x4 pose
        float kx = kf.pose.at<float>(0, 3);
        float ky = kf.pose.at<float>(1, 3);
        float kz = kf.pose.at<float>(2, 3);

        float dx = lx - kx;
        float dy = ly - ky;
        float dz = lz - kz;
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (dist <= radius) {
            within_radius.push_back({&kf, dist});
        }
    }

    // Sort by distance ascending
    std::sort(within_radius.begin(), within_radius.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.dist < b.dist;
              });

    // Return top max_count
    std::vector<KeyframeData*> result;
    int count = std::min(max_count, static_cast<int>(within_radius.size()));
    result.reserve(count);
    for (int i = 0; i < count; i++) {
        result.push_back(within_radius[i].kf);
    }

    return result;
}

// ---------------------------------------------------------------------------
// getGlobalCandidates — compute BoW similarity, return top-K keyframes
// ---------------------------------------------------------------------------
std::vector<KeyframeData*> VisualLocalizer::getGlobalCandidates(
    const cv::Mat& descriptors) {

    std::vector<float> query_bow = computeBoW(descriptors);
    if (query_bow.empty())
        return {};

    struct Candidate {
        KeyframeData* kf;
        float similarity;
    };
    std::vector<Candidate> scored;
    scored.reserve(keyframes_.size());

    for (auto& kf : keyframes_) {
        float sim = cosineSimilarity(query_bow, kf.global_descriptor);
        scored.push_back({&kf, sim});
    }

    // Sort by similarity descending
    std::sort(scored.begin(), scored.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.similarity > b.similarity;
              });

    // Return top-K
    std::vector<KeyframeData*> result;
    int count = std::min(kGlobalTopK, static_cast<int>(scored.size()));
    result.reserve(count);
    for (int i = 0; i < count; i++) {
        result.push_back(scored[i].kf);
    }

    // Record best BoW similarity for debug
    if (!scored.empty()) {
        last_debug_info_.best_bow_sim = scored[0].similarity;
    }

    return result;
}

// ---------------------------------------------------------------------------
// hammingDistance — compare two unsigned char descriptor byte arrays via
//                  XOR + popcount (Brian Kernighan's method).
//                  Both vocabulary and query descriptors are now uint8.
// ---------------------------------------------------------------------------
int VisualLocalizer::hammingDistance(const unsigned char* a, const unsigned char* b,
                                     int len) {
    int dist = 0;
    for (int i = 0; i < len; i++) {
        unsigned char xor_val = a[i] ^ b[i];
        // Brian Kernighan's popcount
        while (xor_val != 0) {
            dist++;
            xor_val &= static_cast<unsigned char>(xor_val - 1);
        }
    }
    return dist;
}

// ---------------------------------------------------------------------------
// cosineSimilarity — dot product of two L2-normalized vectors
// ---------------------------------------------------------------------------
float VisualLocalizer::cosineSimilarity(const std::vector<float>& a,
                                         const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty())
        return 0.0f;

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-9f)
        return 0.0f;

    return dot / denom;
}
