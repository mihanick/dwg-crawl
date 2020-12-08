﻿using System.IO;
using System.Text;
using System;
using System.Diagnostics;

[DebuggerStepThrough]
public class JsonHelper
{
	public static string To<T>(T obj)
	{
		string retVal = null;
		System.Runtime.Serialization.Json.DataContractJsonSerializer serializer = new System.Runtime.Serialization.Json.DataContractJsonSerializer(obj.GetType());
		using (MemoryStream ms = new MemoryStream())
		{
			serializer.WriteObject(ms, obj);

			retVal = Encoding.UTF8.GetString
				(ms.ToArray());
		}

		return retVal;
	}

	public static T From<T>(string json)
	{
		T obj = Activator.CreateInstance<T>();
		using (MemoryStream ms = new MemoryStream(Encoding.Unicode.GetBytes(json)))
		{
			System.Runtime.Serialization.Json.DataContractJsonSerializer serializer = new System.Runtime.Serialization.Json.DataContractJsonSerializer(obj.GetType());
			obj = (T)serializer.ReadObject(ms);
		}

		return obj;
	}
}

