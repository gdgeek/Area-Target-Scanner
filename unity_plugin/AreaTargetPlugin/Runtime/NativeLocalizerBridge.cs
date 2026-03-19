using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace AreaTargetPlugin
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct VLResult
    {
        public int state;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
        public float[] pose;
        public float confidence;
        public int matched_features;
    }

    internal static class NativeLocalizerBridge
    {
        // iOS 静态链接使用 __Internal，其他平台使用动态库名
#if UNITY_IOS && !UNITY_EDITOR
        private const string LibName = "__Internal";
#else
        private const string LibName = "visual_localizer";
#endif

        [DllImport(LibName)] internal static extern IntPtr vl_create();
        [DllImport(LibName)] internal static extern void vl_destroy(IntPtr handle);

        [DllImport(LibName)] internal static extern int vl_add_vocabulary_word(
            IntPtr handle, int word_id, byte[] descriptor, int desc_len, float idf_weight);

        [DllImport(LibName)] internal static extern int vl_add_keyframe(
            IntPtr handle, int kf_id, float[] pose_4x4,
            byte[] descriptors, int desc_count,
            float[] points3d, float[] points2d);

        [DllImport(LibName)] internal static extern int vl_build_index(IntPtr handle);

        [DllImport(LibName)] internal static extern VLResult vl_process_frame(
            IntPtr handle, byte[] image_data, int width, int height,
            float fx, float fy, float cx, float cy,
            int has_last_pose, float[] last_pose_4x4);

        [DllImport(LibName)] internal static extern void vl_reset(IntPtr handle);
    }
}
