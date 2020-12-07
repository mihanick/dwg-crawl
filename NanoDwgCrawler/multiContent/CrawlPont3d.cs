using System;
using System.Runtime.Serialization;

namespace DwgDump.Enitites
{
	[DataContract]
	public class crawlPoint3d
	{
		[DataMember]
		string ClassName = "Point3D";

		[DataMember]
		public double X;
		[DataMember]
		public double Y;
		[DataMember]
		public double Z;

		public crawlPoint3d()
		{
			this.X = 0;
			this.Y = 0;
			this.Z = 0;
		}

		public crawlPoint3d(double X, double Y, double Z)
		{
			this.X = X;
			this.Y = Y;
			this.Z = Z;
		}

		public override string ToString()
		{
			return string.Format("({0}, {1}, {2})", Math.Round(this.X, 2), Math.Round(this.Y, 2), Math.Round(this.Z, 2));
		}

		public bool Equals(crawlPoint3d otherPoint3d)
		{
			return this.ToString().Equals(otherPoint3d.ToString());
		}
	}
}