using System.Threading.Tasks;

namespace AreaTargetPlugin.PointCloudLocalization
{
    public class KalmanDataProcessor : IDataProcessor<SceneUpdateData>
    {
        private readonly KalmanPoseFilter _filter;

        public KalmanDataProcessor(KalmanPoseFilter filter = null)
        {
            _filter = filter ?? new KalmanPoseFilter();
        }

        public Task<SceneUpdateData> ProcessData(SceneUpdateData data, DataProcessorTrigger trigger)
        {
            data.Pose = _filter.Update(data.Pose);
            return Task.FromResult(data);
        }

        public Task ResetProcessor()
        {
            _filter.Reset();
            return Task.CompletedTask;
        }
    }
}
