namespace AreaTargetPlugin
{
    /// <summary>
    /// Represents the current tracking state of the area target tracker.
    /// </summary>
    public enum TrackingState
    {
        /// <summary>Asset bundle loaded, tracker is initializing.</summary>
        INITIALIZING,

        /// <summary>Actively tracking with a valid pose.</summary>
        TRACKING,

        /// <summary>Tracking lost, attempting to relocalize.</summary>
        LOST
    }
}
