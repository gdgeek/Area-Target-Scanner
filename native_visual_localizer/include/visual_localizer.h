/**
 * visual_localizer.h - C API for Visual Localizer native plugin
 *
 * Provides ORB feature extraction, BoW keyframe retrieval, feature matching,
 * and PnP RANSAC localization via an opaque handle interface.
 * Designed for P/Invoke from Unity C#.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #define VL_API __declspec(dllexport)
#else
  #define VL_API __attribute__((visibility("default")))
#endif

typedef void* VLHandle;

typedef struct {
    int state;           /* 0=INITIALIZING, 1=TRACKING, 2=LOST */
    float pose[16];      /* 4x4 row-major transform matrix */
    float confidence;    /* [0,1] */
    int matched_features;
} VLResult;

/* Lifecycle */
VL_API VLHandle vl_create(void);
VL_API void     vl_destroy(VLHandle handle);

/* Data loading (called from C# after SQLite read) */
VL_API int  vl_add_vocabulary_word(VLHandle handle, int word_id,
                                    const unsigned char* descriptor, int desc_len,
                                    float idf_weight);
VL_API int  vl_add_keyframe(VLHandle handle, int kf_id,
                             const float* pose_4x4,
                             const unsigned char* descriptors, int desc_count,
                             const float* points3d, const float* points2d);
VL_API int  vl_build_index(VLHandle handle);

/* Frame processing */
VL_API VLResult vl_process_frame(VLHandle handle,
                                  const unsigned char* image_data,
                                  int width, int height,
                                  float fx, float fy, float cx, float cy,
                                  int has_last_pose, const float* last_pose_4x4);

/* State management */
VL_API void vl_reset(VLHandle handle);

#ifdef __cplusplus
}
#endif
