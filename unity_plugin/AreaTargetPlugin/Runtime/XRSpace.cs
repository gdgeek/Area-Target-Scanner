using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace AreaTargetPlugin.PointCloudLocalization
{
    public class XRSpace : MonoBehaviour, ISceneUpdateable
    {
        public bool ProcessPoses { get; set; } = true;
        private List<IDataProcessor<SceneUpdateData>> _processors = new List<IDataProcessor<SceneUpdateData>>();

        public Transform GetTransform() => transform;

        public async Task SceneUpdate(SceneUpdateData data)
        {
            if (data.Ignore) return;

            if (ProcessPoses)
            {
                foreach (var p in _processors)
                    data = await p.ProcessData(data, DataProcessorTrigger.NewData);
            }

            Vector3 position = new Vector3(data.Pose.m03, data.Pose.m13, data.Pose.m23);
            Quaternion rotation = data.Pose.rotation;
            transform.SetPositionAndRotation(position, rotation);
        }

        public async Task ResetScene()
        {
            foreach (var p in _processors)
                await p.ResetProcessor();
        }

        public void AddProcessor(IDataProcessor<SceneUpdateData> processor)
            => _processors.Add(processor);

        private async void OnDestroy()
        {
            foreach (var p in _processors)
                await p.ResetProcessor();
        }
    }
}
