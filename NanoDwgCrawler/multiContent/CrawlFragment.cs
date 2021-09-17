using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace DwgDump.Enitites
{
	[DataContract]
	public class CrawlFragment
	{
		[DataMember]
		public string FileId;

		[DataMember]
		public string GroupId;

		[DataMember]
		public string StrippedFileName;

		[DataMember]
		public string AnnotatedFileName;

		[DataMember]
		public CrawlPoint3d MinBoundPoint;

		[DataMember]
		public CrawlPoint3d MaxBoundPoint;
	}
}
