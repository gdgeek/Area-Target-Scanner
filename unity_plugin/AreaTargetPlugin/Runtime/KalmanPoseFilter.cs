using System;
using UnityEngine;

namespace AreaTargetPlugin
{
    /// <summary>
    /// Kalman filter for 6DoF pose smoothing.
    /// State vector: [tx, ty, tz, rx, ry, rz] (translation + Euler angles).
    /// Implements Algorithm 5 from the design document.
    /// Validates: Requirement 13.1
    /// </summary>
    public class KalmanPoseFilter
    {
        private const int StateSize = 6;

        // State vector [tx, ty, tz, rx, ry, rz]
        private float[] _x;
        // Covariance matrix (StateSize x StateSize, row-major)
        private float[] _P;
        // State transition matrix (identity for constant model)
        private readonly float[] _F;
        // Observation matrix (identity — we observe all states directly)
        private readonly float[] _H;
        // Process noise covariance
        private readonly float[] _Q;
        // Measurement noise covariance
        private readonly float[] _R;

        private bool _initialized;

        /// <summary>
        /// Creates a new Kalman pose filter with default noise parameters.
        /// </summary>
        /// <param name="processNoise">Process noise standard deviation (default 0.01).</param>
        /// <param name="measurementNoise">Measurement noise standard deviation (default 0.1).</param>
        public KalmanPoseFilter(float processNoise = 0.01f, float measurementNoise = 0.1f)
        {
            _x = new float[StateSize];
            _P = CreateIdentityScaled(StateSize, 1.0f);
            _F = CreateIdentity(StateSize);
            _H = CreateIdentity(StateSize);
            _Q = CreateIdentityScaled(StateSize, processNoise * processNoise);
            _R = CreateIdentityScaled(StateSize, measurementNoise * measurementNoise);
            _initialized = false;
        }

        /// <summary>
        /// Whether the filter has been initialized with at least one measurement.
        /// </summary>
        public bool IsInitialized => _initialized;

        /// <summary>
        /// Resets the filter state, requiring re-initialization on next update.
        /// </summary>
        public void Reset()
        {
            _x = new float[StateSize];
            _P = CreateIdentityScaled(StateSize, 1.0f);
            _initialized = false;
        }

        /// <summary>
        /// Processes a raw pose through the Kalman filter and returns the smoothed pose.
        /// </summary>
        /// <param name="rawPose">The raw 4x4 pose matrix from PnP.</param>
        /// <returns>The smoothed 4x4 pose matrix.</returns>
        public Matrix4x4 Update(Matrix4x4 rawPose)
        {
            // Decompose pose into translation + Euler angles
            float[] measurement = PoseToState(rawPose);

            if (!_initialized)
            {
                // First measurement: initialize state directly
                Array.Copy(measurement, _x, StateSize);
                _initialized = true;
                return rawPose;
            }

            // --- Predict step ---
            // x_pred = F * x (F is identity, so x_pred = x)
            float[] xPred = MatVecMul(_F, _x);
            // P_pred = F * P * F^T + Q (F is identity, so P_pred = P + Q)
            float[] pPred = MatAdd(MatMulTranspose(_F, _P), _Q);

            // --- Update step ---
            // Innovation: y = measurement - H * x_pred
            float[] hx = MatVecMul(_H, xPred);
            float[] innovation = new float[StateSize];
            for (int i = 0; i < StateSize; i++)
                innovation[i] = measurement[i] - hx[i];

            // Wrap Euler angle innovations to [-pi, pi]
            for (int i = 3; i < StateSize; i++)
                innovation[i] = WrapAngle(innovation[i]);

            // S = H * P_pred * H^T + R
            float[] s = MatAdd(MatMulTranspose(_H, MatMul(_H, pPred)), _R);

            // K = P_pred * H^T * S^-1
            float[] sInv = MatInverse(s);
            float[] k = MatMul(MatMulTranspose(pPred, _H), sInv);

            // x = x_pred + K * innovation
            float[] kInnovation = MatVecMul(k, innovation);
            for (int i = 0; i < StateSize; i++)
                _x[i] = xPred[i] + kInnovation[i];

            // P = (I - K * H) * P_pred
            float[] kh = MatMul(k, _H);
            float[] iMinusKH = MatSubtractFromIdentity(kh);
            _P = MatMul(iMinusKH, pPred);

            // Reconstruct smoothed pose matrix
            return StateToPose(_x);
        }

        /// <summary>
        /// Decomposes a 4x4 pose matrix into [tx, ty, tz, rx, ry, rz].
        /// </summary>
        internal static float[] PoseToState(Matrix4x4 pose)
        {
            float[] state = new float[StateSize];
            // Translation
            state[0] = pose.m03;
            state[1] = pose.m13;
            state[2] = pose.m23;
            // Euler angles from rotation matrix
            float[] euler = MatrixToEuler(pose);
            state[3] = euler[0];
            state[4] = euler[1];
            state[5] = euler[2];
            return state;
        }

        /// <summary>
        /// Reconstructs a 4x4 pose matrix from state [tx, ty, tz, rx, ry, rz].
        /// </summary>
        internal static Matrix4x4 StateToPose(float[] state)
        {
            Matrix4x4 rot = EulerToMatrix(state[3], state[4], state[5]);
            Matrix4x4 pose = rot;
            pose.m03 = state[0];
            pose.m13 = state[1];
            pose.m23 = state[2];
            pose.m30 = 0f;
            pose.m31 = 0f;
            pose.m32 = 0f;
            pose.m33 = 1f;
            return pose;
        }

        /// <summary>
        /// Extracts Euler angles (rx, ry, rz) from a rotation matrix using ZYX convention.
        /// </summary>
        internal static float[] MatrixToEuler(Matrix4x4 m)
        {
            float sy = -m.m20;
            float ry, rx, rz;

            if (Mathf.Abs(sy) < 0.99999f)
            {
                ry = Mathf.Asin(sy);
                rx = Mathf.Atan2(m.m21, m.m22);
                rz = Mathf.Atan2(m.m10, m.m00);
            }
            else
            {
                // Gimbal lock
                ry = sy > 0 ? Mathf.PI / 2f : -Mathf.PI / 2f;
                rx = Mathf.Atan2(-m.m12, m.m11);
                rz = 0f;
            }

            return new float[] { rx, ry, rz };
        }

        /// <summary>
        /// Constructs a rotation matrix from Euler angles (rx, ry, rz) using ZYX convention.
        /// </summary>
        internal static Matrix4x4 EulerToMatrix(float rx, float ry, float rz)
        {
            float cx = Mathf.Cos(rx), sx = Mathf.Sin(rx);
            float cy = Mathf.Cos(ry), sy = Mathf.Sin(ry);
            float cz = Mathf.Cos(rz), sz = Mathf.Sin(rz);

            Matrix4x4 m = new Matrix4x4();
            m.m00 = cy * cz;
            m.m01 = sx * sy * cz - cx * sz;
            m.m02 = cx * sy * cz + sx * sz;
            m.m10 = cy * sz;
            m.m11 = sx * sy * sz + cx * cz;
            m.m12 = cx * sy * sz - sx * cz;
            m.m20 = -sy;
            m.m21 = sx * cy;
            m.m22 = cx * cy;
            m.m30 = 0f; m.m31 = 0f; m.m32 = 0f; m.m33 = 1f;
            m.m03 = 0f; m.m13 = 0f; m.m23 = 0f;
            return m;
        }

        /// <summary>
        /// Wraps an angle to the range [-PI, PI].
        /// </summary>
        private static float WrapAngle(float angle)
        {
            while (angle > Mathf.PI) angle -= 2f * Mathf.PI;
            while (angle < -Mathf.PI) angle += 2f * Mathf.PI;
            return angle;
        }

        #region Matrix Utilities (NxN row-major float arrays)

        private static float[] CreateIdentity(int n)
        {
            float[] m = new float[n * n];
            for (int i = 0; i < n; i++) m[i * n + i] = 1f;
            return m;
        }

        private static float[] CreateIdentityScaled(int n, float scale)
        {
            float[] m = new float[n * n];
            for (int i = 0; i < n; i++) m[i * n + i] = scale;
            return m;
        }

        /// <summary>Matrix-vector multiply: result = M * v</summary>
        private static float[] MatVecMul(float[] M, float[] v)
        {
            int n = v.Length;
            float[] result = new float[n];
            for (int i = 0; i < n; i++)
            {
                float sum = 0f;
                for (int j = 0; j < n; j++)
                    sum += M[i * n + j] * v[j];
                result[i] = sum;
            }
            return result;
        }

        /// <summary>Matrix add: C = A + B</summary>
        private static float[] MatAdd(float[] A, float[] B)
        {
            float[] C = new float[A.Length];
            for (int i = 0; i < A.Length; i++) C[i] = A[i] + B[i];
            return C;
        }

        /// <summary>Matrix multiply: C = A * B (both NxN)</summary>
        private static float[] MatMul(float[] A, float[] B)
        {
            int n = (int)Mathf.Sqrt(A.Length);
            float[] C = new float[n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < n; k++)
                        sum += A[i * n + k] * B[k * n + j];
                    C[i * n + j] = sum;
                }
            return C;
        }

        /// <summary>Computes A * B^T for NxN matrices.</summary>
        private static float[] MatMulTranspose(float[] A, float[] B)
        {
            int n = (int)Mathf.Sqrt(A.Length);
            float[] C = new float[n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < n; k++)
                        sum += A[i * n + k] * B[j * n + k]; // B transposed
                    C[i * n + j] = sum;
                }
            return C;
        }

        /// <summary>Computes I - M for NxN matrix.</summary>
        private static float[] MatSubtractFromIdentity(float[] M)
        {
            int n = (int)Mathf.Sqrt(M.Length);
            float[] result = new float[M.Length];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    float identity = (i == j) ? 1f : 0f;
                    result[i * n + j] = identity - M[i * n + j];
                }
            return result;
        }

        /// <summary>
        /// Inverts a 6x6 matrix using Gauss-Jordan elimination.
        /// </summary>
        private static float[] MatInverse(float[] M)
        {
            int n = (int)Mathf.Sqrt(M.Length);
            // Augmented matrix [M | I]
            float[,] aug = new float[n, 2 * n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                    aug[i, j] = M[i * n + j];
                aug[i, n + i] = 1f;
            }

            for (int col = 0; col < n; col++)
            {
                // Partial pivoting
                int maxRow = col;
                float maxVal = Mathf.Abs(aug[col, col]);
                for (int row = col + 1; row < n; row++)
                {
                    float val = Mathf.Abs(aug[row, col]);
                    if (val > maxVal) { maxVal = val; maxRow = row; }
                }
                if (maxRow != col)
                {
                    for (int j = 0; j < 2 * n; j++)
                    {
                        float tmp = aug[col, j];
                        aug[col, j] = aug[maxRow, j];
                        aug[maxRow, j] = tmp;
                    }
                }

                float pivot = aug[col, col];
                if (Mathf.Abs(pivot) < 1e-12f)
                {
                    // Singular — return identity as fallback
                    return CreateIdentity(n);
                }

                for (int j = 0; j < 2 * n; j++)
                    aug[col, j] /= pivot;

                for (int row = 0; row < n; row++)
                {
                    if (row == col) continue;
                    float factor = aug[row, col];
                    for (int j = 0; j < 2 * n; j++)
                        aug[row, j] -= factor * aug[col, j];
                }
            }

            float[] inv = new float[n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    inv[i * n + j] = aug[i, n + j];
            return inv;
        }

        #endregion
    }
}
