using System.Runtime.Serialization;

using System.Collections.Generic;
using System;
using DwgDump.Util;
using Multicad.DatabaseServices;
using Multicad.DatabaseServices.StandardObjects;
using Multicad.Dimensions;
using Multicad.Geometry;
using System.Linq;
using Multicad.Symbols;

namespace DwgDump.Enitites
{
	[DataContract]
	public class CrawlEntity
	{
		[DataMember]
		public string ObjectId;

		[DataMember]
		public string Color;
		[DataMember]
		public string Layer;
		[DataMember]
		public string Linetype;
		[DataMember]
		public string LineWeight;
	}

	[DataContract]
	public class CrawlLine : CrawlEntity
	{
		[DataMember]
		public string ClassName = "Line";

		[DataMember]
		public CrawlPoint3d EndPoint { get; set; }
		[DataMember]
		public CrawlPoint3d StartPoint { get; set; }

		public CrawlLine()
		{
		}

		public CrawlLine(CrawlPoint3d start, CrawlPoint3d end)
		{
			this.StartPoint = start;
			this.EndPoint = end;
		}
	}

	[DataContract]
	public class CrawlPolyline : CrawlEntity
	{
		[DataMember]
		public string ClassName = "Polyline";

		[DataMember]
		public List<CrawlPoint3d> Vertices = new List<CrawlPoint3d>();
	}

	[DataContract]
	public class CrawlText : CrawlEntity
	{
		[DataMember]
		public string ClassName = "Text";

		[DataMember]
		public CrawlPoint3d Position { get; set; }
		[DataMember]
		public string TextString;
	}

	[DataContract]
	public class CrawlArc : CrawlEntity
	{
		[DataMember]
		public string ClassName = "Arc";

		[DataMember]
		public CrawlPoint3d Center { get; set; }
		[DataMember]
		public CrawlPoint3d StartPoint { get; set; }
		[DataMember]
		public CrawlPoint3d EndPoint { get; set; }

		[DataMember]
		public double Thickness;
		[DataMember]
		public double Radius;

	}

	[DataContract]
	public class CrawlCircle : CrawlEntity
	{
		[DataMember]
		public string ClassName = "Circle";

		[DataMember]
		public CrawlPoint3d Center { get; set; }
		[DataMember]
		public CrawlPoint3d StartPoint { get; set; }
		[DataMember]
		public CrawlPoint3d EndPoint { get; set; }

		[DataMember]
		public double Radius;

	}

	[DataContract]
	public class Crawlellipse : CrawlEntity
	{
		[DataMember]
		public string ClassName = "Ellipse";

		[DataMember]
		public CrawlPoint3d Center { get; set; }
		[DataMember]
		public CrawlPoint3d StartPoint { get; set; }
		[DataMember]
		public CrawlPoint3d EndPoint { get; set; }
		[DataMember]
		public CrawlPoint3d MajorAxisVector { get; set; }
		[DataMember]
		public CrawlPoint3d MinorAxisVector { get; set; }
	}

	[DataContract]
	public class LinearDimension : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AlignedDimension";

		[DataMember]
		public CrawlPoint3d XLine1Point { get; set; }
		[DataMember]
		public CrawlPoint3d XLine2Point { get; set; }
		[DataMember]
		public CrawlPoint3d DimLinePoint { get; set; }
		[DataMember]
		public CrawlPoint3d TextPosition { get; set; }

		[DataMember]
		public string DimensionText;
		[DataMember]
		public string DimensionStyleName;
	}

	[DataContract]
	public class AngularDimension : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AngularDimension";

		[DataMember]
		public CrawlPoint3d XLine1Point { get; set; }
		[DataMember]
		public CrawlPoint3d XLine2Point { get; set; }
		[DataMember]
		public CrawlPoint3d CenterPoint { get; set; }
		[DataMember]
		public CrawlPoint3d TextPosition { get; set; }

		[DataMember]
		public string DimensionText;
		[DataMember]
		public string DimensionStyleName;
	}

	[DataContract]
	public class DiameterDimension : CrawlEntity
	{
		[DataMember]
		public string ClassName = "DiametricDimension";

		[DataMember]
		public double ArcStartAngle { get; set; }
		[DataMember]
		public double ArcEndAngle { get; set; }
		[DataMember]
		public CrawlPoint3d Center { get; set; }
		[DataMember]
		public CrawlPoint3d Pos1 { get; set; }
		[DataMember]
		public CrawlPoint3d TextPosition { get; set; }

		[DataMember]
		public string DimensionText;
		[DataMember]
		public string DimensionStyleName;
	}

	[DataContract]
	public class ArcDimension : CrawlEntity
	{
		[DataMember]
		public string ClassName = "ArcDimension";

		[DataMember]
		public double Radius { get; set; }
		[DataMember]
		public CrawlPoint3d Center { get; set; }

		[DataMember]
		public CrawlPoint3d TextPosition { get; set; }

		[DataMember]
		public string DimensionText;
		[DataMember]
		public string DimensionStyleName;
	}

	[DataContract]
	public class RadialDimension : CrawlEntity
	{
		[DataMember]
		public string ClassName = "RadialDimension";
		[DataMember]
		public double ArcEndAngle { get; set; }
		[DataMember]
		public double ArcStartAngle { get; set; }
		[DataMember]
		public double Radius { get; set; }
		[DataMember]
		public CrawlPoint3d Center { get; set; }
		[DataMember]
		public CrawlPoint3d Position { get; set; }

		[DataMember]
		public CrawlPoint3d TextPosition { get; set; }

		[DataMember]
		public string DimensionText;
		[DataMember]
		public string DimensionStyleName;
	}

	[DataContract]
	public class CrawlHatch : CrawlEntity
	{
		[DataMember]
		public string ClassName = "Hatch";

		[DataMember]
		public string PatternName;

		[DataMember]
		public List<CrawlPolyline> Loops = new List<CrawlPolyline>();
	}

	[DataContract]
	public class Spline : CrawlEntity
	{
		[DataMember]
		public string ClassName = "Spline";

		[DataMember]
		public List<CrawlPoint3d> Vertices = new List<CrawlPoint3d>();
		[DataMember]
		public List<CrawlPoint3d> ControlPoints = new List<CrawlPoint3d>();
	}

	[DataContract]
	public class CrawlAcDbPoint : CrawlEntity
	{
		[DataMember]
		public string ClassName = "Point";

		[DataMember]
		public CrawlPoint3d Position { get; set; }
	}

	[DataContract]
	public class CrawlAcDbBlockReference : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbBlockReference";

		[DataMember]
		public CrawlPoint3d Position;

		[DataMember]
		public string Name;

	}

	[DataContract]
	public class Section : CrawlPolyline
	{
		[DataMember]
		public string ClassName = "SectionDesignation";

		[DataMember]
		public string Name;
	}

	[DataContract]
	public class AxisLinear : CrawlLine
	{
		[DataMember]
		public string ClassName = "Axis";

		[DataMember]
		public string Name;
	}

	/* TODO: Not implemented
	[DataContract]
	public class CrawlAcDbProxyEntity : CrawlEntity
	{
		[DataMember]
		public string ClassName = "ProxyEntity";

		[DataMember]
		public string BlockId;
		[DataMember]
		public string FileId;

		public CrawlAcDbProxyEntity(DbProxyEntity prxy)
		{
			Entity ent = (Entity)prxy;
			this.ObjectId = ent.ObjectId.ToString();

			this.Layer = prxy.Layer;
			this.Linetype = prxy.Linetype;
			this.LineWeight = prxy.LineWeight.ToString();
			this.Color = prxy.Color.ToString();
		}
	}

	[DataContract]
	public class CrawlAcDbBlockTableRecord
	{
		[DataMember]
		public string ClassName = "AcDbBlockTableRecord";


		[DataMember]
		public string Name;
		[DataMember]
		public string FilePath;

		[DataMember]
		public string FileId;
		[DataMember]
		public string BlockId;
		[DataMember]
		public string ObjectId;

		public CrawlAcDbBlockTableRecord() { }

		public CrawlAcDbBlockTableRecord(BlockTableRecord btr, string filePath)
		{
			this.Name = btr.Name;
			this.FilePath = filePath;
			this.ObjectId = btr.ObjectId.ToString();
		}
	}

	[DataContract]
	public class CrawlAcDbSolid : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbSolid";

		[DataMember]
		public string FileId;

		[DataMember]
		public string ParentFileId;

		[DataMember]
		public List<CrawlPoint3d> Vertices;

		public CrawlAcDbSolid() { }

		public CrawlAcDbSolid(Solid solid)
		{
			Entity ent = (Entity)solid;
			this.ObjectId = ent.ObjectId.ToString();

			this.Layer = solid.Layer;
			this.Linetype = solid.Linetype;
			this.LineWeight = solid.LineWeight.ToString();
			this.Color = solid.Color.ToString();

			Vertices = new List<CrawlPoint3d>();
			short i = 0;
			Point3d pt = solid.GetPointAt(i);
			try
			{
				while (pt != null)
				{
					Vertices.Add(new CrawlPoint3d(pt.X, pt.Y, pt.Z));
					i++;
					pt = solid.GetPointAt(i);
				}
			}
			catch { }

		}
	}
	*/

	[DataContract]
	public class CrawlAcDbLayerTableRecord
	{
		[DataMember]
		public string ClassName = "AcDbLayerTableRecord";

		[DataMember]
		public string Name;
		[DataMember]
		public string Linetype;
		[DataMember]
		public bool IsFrozen;
		[DataMember]
		public bool IsHidden;
		[DataMember]
		public bool IsOff;
		[DataMember]
		public bool IsPlottable;
		[DataMember]
		public string LineWeight;
		[DataMember]
		public string Color;
		[DataMember]
		public string ObjectId;
	}

	[DataContract]
	public class CrawlNote : CrawlEntity
	{
		[DataMember]
		public string SecondLine { get; set; }
		[DataMember]
		public string FirstLine { get; set; }
		[DataMember]
		public CrawlPoint3d Origin { get; set; }
		[DataMember]
		public List<CrawlLine> Lines = new List<CrawlLine>();
	}

	[DataContract]
	public class Break : CrawlLine
	{
	}
}