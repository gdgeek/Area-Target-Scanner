using System.Threading.Tasks;
using UnityEngine;

namespace AreaTargetPlugin.PointCloudLocalization
{
    public interface ISceneUpdateable
    {
        Transform GetTransform();
        Task SceneUpdate(SceneUpdateData data);
        Task ResetScene();
    }
}
