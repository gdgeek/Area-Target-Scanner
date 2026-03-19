using System.Collections.Generic;
using UnityEngine;

namespace AreaTargetPlugin.PointCloudLocalization
{
    public static class MapManager
    {
        private static readonly Dictionary<int, MapEntry> _maps = new Dictionary<int, MapEntry>();

        public static void RegisterMap(int mapId, MapEntry entry)
        {
            if (_maps.ContainsKey(mapId))
            {
                Debug.LogWarning($"[MapManager] Overwriting existing map entry for mapId: {mapId}");
            }
            _maps[mapId] = entry;
        }

        public static void UnregisterMap(int mapId)
        {
            _maps.Remove(mapId);
        }

        public static bool TryGetMapEntry(int mapId, out MapEntry entry)
        {
            return _maps.TryGetValue(mapId, out entry);
        }

        public static void Clear()
        {
            _maps.Clear();
        }
    }
}
