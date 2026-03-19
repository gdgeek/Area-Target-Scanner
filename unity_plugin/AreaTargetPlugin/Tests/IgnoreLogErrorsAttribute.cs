using System;
using NUnit.Framework;
using NUnit.Framework.Interfaces;
using UnityEngine.TestTools;

namespace AreaTargetPlugin.Tests
{
    /// <summary>
    /// Custom NUnit attribute that suppresses Debug.LogError failures in tests.
    /// Apply to a test class to ignore all error log messages.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Method)]
    public class IgnoreLogErrorsAttribute : NUnitAttribute, ITestAction
    {
        public ActionTargets Targets => ActionTargets.Test;

        public void BeforeTest(ITest test)
        {
            LogAssert.ignoreFailingMessages = true;
        }

        public void AfterTest(ITest test)
        {
        }
    }
}
