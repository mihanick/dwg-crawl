

using System.Runtime.Serialization;
using System.Collections.Generic;

namespace DwgDump.Enitites
{
	[DataContract]
	public abstract class CrawlEntity
	{
		[DataMember]
		public string ClassName = "CrawlEntity";
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

		[DataMember]
		public string FileId;
		[DataMember]
		public string GroupId;
		[DataMember]
		public string BlockId;
		[DataMember]
		public long Handle;

		public void Copy(CrawlEntity ent)
		{
			// Set common properties

			if (ent != null)
			{
				Layer = ent.Layer;
				ObjectId = ent.ObjectId;
				Linetype = ent.Linetype;
				LineWeight = ent.LineWeight;
				Color = ent.Color;

				FileId = ent.FileId;
				GroupId = ent.GroupId;
				BlockId = ent.BlockId;
			}
		}

		public override string ToString()
		{
			return ClassName;
		}
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
	public class BlockReference : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbBlockReference";

		[DataMember]
		public CrawlPoint3d Position;

		[DataMember]
		public string Name;

		public List<CrawlEntity> Contents = new List<CrawlEntity>();
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

	[DataContract]
	public class CrawlNote : CrawlEntity
	{
		[DataMember]
		public string ClassName = "Note";
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
		[DataMember]
		public string ClassName = "Break";
	}

	[DataContract]
	public class LayerTableRecord : CrawlEntity
	{
		[DataMember]
		public string ClassName = "LayerTableRecord";

		[DataMember]
		public string Name;

		[DataMember]
		public bool IsFrozen;
		[DataMember]
		public bool IsHidden;
		[DataMember]
		public bool IsOff;
		[DataMember]
		public bool IsPlottable;
	}
}