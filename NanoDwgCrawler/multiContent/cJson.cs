using System.IO;
using System.Text;
using System;
using System.Diagnostics;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

[DebuggerStepThrough]
public class CrawlJsonHelper
{
	public static string Serialize(object o, string className = "")
	{
		var jsn = JsonConvert.SerializeObject(o);
		JObject j = JObject.Parse(jsn);

		if (!string.IsNullOrEmpty(className))
			j["ClassName"] = className;

		return j.ToString();
	}
}