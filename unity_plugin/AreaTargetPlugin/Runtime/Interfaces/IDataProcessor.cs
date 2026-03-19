using System.Threading.Tasks;

namespace AreaTargetPlugin.PointCloudLocalization
{
    public enum DataProcessorTrigger
    {
        NewData,
        Update
    }

    public interface IDataProcessor<T>
    {
        Task<T> ProcessData(T data, DataProcessorTrigger trigger);
        Task ResetProcessor();
    }
}
