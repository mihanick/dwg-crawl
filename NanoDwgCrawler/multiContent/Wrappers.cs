using System.Runtime.Serialization;
using Teigha.Geometry;
using Teigha.DatabaseServices;
using System.Collections.Generic;
using System;
using DwgDump.Util;

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
	public class CrawlAcDbLine : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbLine";

		[DataMember]
		public CrawlPoint3d EndPoint { get; set; }
		[DataMember]
		public CrawlPoint3d StartPoint { get; set; }
		[DataMember]
		public double Length;

		public CrawlAcDbLine()
		{
		}

		public CrawlAcDbLine(Line line)
		{
			Entity ent = (Entity)line;
			this.ObjectId = ent.ObjectId.ToString();

			this.EndPoint = new CrawlPoint3d(line.EndPoint.X, line.EndPoint.Y, line.EndPoint.Z);
			this.StartPoint = new CrawlPoint3d(line.StartPoint.X, line.StartPoint.Y, line.StartPoint.Z);
			this.Layer = line.Layer;
			this.Linetype = line.Linetype;
			this.LineWeight = line.LineWeight.ToString();
			this.Color = line.Color.ToString();

			this.Length = line.Length;
		}
	}

	[DataContract]
	public class CrawlAcDbPolyline : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbPolyline";

		[DataMember]
		public double Length;
		[DataMember]
		public double Area;

		[DataMember]
		public List<CrawlPoint3d> Vertixes;

		public CrawlAcDbPolyline() { }

		public CrawlAcDbPolyline(Polyline polyline)
		{
			Entity ent = (Entity)polyline;
			this.ObjectId = ent.ObjectId.ToString();

			this.Length = polyline.Length;
			this.Area = polyline.Area;

			this.Layer = polyline.Layer;
			this.Linetype = polyline.Linetype;
			this.LineWeight = polyline.LineWeight.ToString();
			this.Color = polyline.Color.ToString();

			Vertixes = new List<CrawlPoint3d>();

			// Use a for loop to get each vertex, one by one
			int vn = polyline.NumberOfVertices;
			for (int i = 0; i < vn; i++)
			{
				double x = polyline.GetPoint3dAt(i).X;
				double y = polyline.GetPoint3dAt(i).Y;
				double z = polyline.GetPoint3dAt(i).Z;
				Vertixes.Add(new CrawlPoint3d(x, y, z));
			}
		}

		public CrawlAcDbPolyline(Polyline2d polyline)
		{
			Entity ent = (Entity)polyline;
			this.ObjectId = ent.ObjectId.ToString();

			Length = polyline.Length;
			this.Layer = polyline.Layer;
			this.Linetype = polyline.Linetype;
			this.LineWeight = polyline.LineWeight.ToString();
			this.Color = polyline.Color.ToString();

			Vertixes = new List<CrawlPoint3d>();

			// Use foreach to get each contained vertex
			foreach (ObjectId vId in polyline)
			{
				Vertex2d v2d =
				  (Vertex2d)
					vId.GetObject(
					OpenMode.ForRead
				  );
				double x = v2d.Position.X;
				double y = v2d.Position.Y;
				double z = v2d.Position.Z;
				Vertixes.Add(new CrawlPoint3d(x, y, z));
			}
		}

		public CrawlAcDbPolyline(Polyline3d polyline)
		{
			Entity ent = (Entity)polyline;
			this.ObjectId = ent.ObjectId.ToString();

			Length = polyline.Length;
			this.Layer = polyline.Layer;
			this.Linetype = polyline.Linetype;
			this.LineWeight = polyline.LineWeight.ToString();
			this.Color = polyline.Color.ToString();

			Vertixes = new List<CrawlPoint3d>();

			// Use foreach to get each contained vertex
			foreach (ObjectId vId in polyline)
			{
				PolylineVertex3d v3d =
				  (PolylineVertex3d)
					vId.GetObject(OpenMode.ForRead);
				double x = v3d.Position.X;
				double y = v3d.Position.Y;
				double z = v3d.Position.Z;
				Vertixes.Add(new CrawlPoint3d(x, y, z));
			}
		}
	}

	[DataContract]
	public class CrawlAcDbText : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbText";

		[DataMember]
		public CrawlPoint3d Position { get; set; }
		[DataMember]
		public string TextString;

		public CrawlAcDbText() { }

		public CrawlAcDbText(DBText text)
		{
			Entity ent = (Entity)text;
			this.ObjectId = ent.ObjectId.ToString();

			this.Position = new CrawlPoint3d(text.Position.X, text.Position.Y, text.Position.Z);

			this.Layer = text.Layer;
			this.Linetype = text.Linetype;
			this.LineWeight = text.LineWeight.ToString();
			this.Color = text.Color.ToString();

			this.TextString = text.TextString;
		}
	}

	[DataContract]
	public class CrawlAcDbMText : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbMText";

		[DataMember]
		public CrawlPoint3d Position { get; set; }
		[DataMember]
		public string TextString;

		public CrawlAcDbMText() { }

		public CrawlAcDbMText(MText text)
		{
			Entity ent = (Entity)text;
			this.ObjectId = ent.ObjectId.ToString();

			this.Position = new CrawlPoint3d(text.Location.X, text.Location.Y, text.Location.Z);
			this.Layer = text.Layer;
			this.Linetype = text.Linetype;
			this.LineWeight = text.LineWeight.ToString();
			this.Color = text.Color.ToString();

			this.TextString = text.Contents;
		}
	}

	[DataContract]
	public class CrawlAcDbAttributeDefinition : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbAttributeDefinition";

		[DataMember]
		public CrawlPoint3d Position { get; set; }
		[DataMember]
		public string TextString;

		[DataMember]
		public string Prompt;
		[DataMember]
		public string Tag;

		[DataMember]
		public CrawlAcDbMText MTextAttributeDefinition;

		public CrawlAcDbAttributeDefinition() { }

		public CrawlAcDbAttributeDefinition(AttributeDefinition att)
		{
			Entity ent = (Entity)att;
			this.ObjectId = ent.ObjectId.ToString();

			this.Position = new CrawlPoint3d(att.Position.X, att.Position.Y, att.Position.Z);
			this.Layer = att.Layer;
			this.Linetype = att.Linetype;
			this.LineWeight = att.LineWeight.ToString();
			this.Color = att.Color.ToString();

			this.Prompt = att.Prompt;
			this.Tag = att.Tag;

			if (att.MTextAttributeDefinition != null)
				this.MTextAttributeDefinition = new CrawlAcDbMText(att.MTextAttributeDefinition);
		}
	}

	[DataContract]
	public class CrawlAcDbArc : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbArc";

		[DataMember]
		public CrawlPoint3d Center { get; set; }
		[DataMember]
		public CrawlPoint3d StartPoint { get; set; }
		[DataMember]
		public CrawlPoint3d EndPoint { get; set; }

		[DataMember]
		public double Length;
		[DataMember]
		public double Thickness;
		[DataMember]
		public double Radius;

		public CrawlAcDbArc() { }

		public CrawlAcDbArc(Arc arc)
		{
			Entity ent = (Entity)arc;
			this.ObjectId = ent.ObjectId.ToString();

			this.EndPoint = new CrawlPoint3d(arc.EndPoint.X, arc.EndPoint.Y, arc.EndPoint.Z);
			this.StartPoint = new CrawlPoint3d(arc.StartPoint.X, arc.StartPoint.Y, arc.StartPoint.Z);
			this.Center = new CrawlPoint3d(arc.Center.X, arc.Center.Y, arc.Center.Z);

			this.Layer = arc.Layer;
			this.Linetype = arc.Linetype;
			this.LineWeight = arc.LineWeight.ToString();
			this.Color = arc.Color.ToString();

			this.Length = arc.Length;
			this.Thickness = arc.Thickness;

			this.Radius = arc.Radius;
		}
	}

	[DataContract]
	public class CrawlAcDbCircle : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbCircle";

		[DataMember]
		public CrawlPoint3d Center { get; set; }
		[DataMember]
		public CrawlPoint3d StartPoint { get; set; }
		[DataMember]
		public CrawlPoint3d EndPoint { get; set; }

		[DataMember]
		public double Length;
		[DataMember]
		public double Thickness;
		[DataMember]
		public double Radius;

		public CrawlAcDbCircle() { }

		public CrawlAcDbCircle(Circle circle)
		{
			Entity ent = (Entity)circle;
			this.ObjectId = ent.ObjectId.ToString();

			this.EndPoint = new CrawlPoint3d(circle.EndPoint.X, circle.EndPoint.Y, circle.EndPoint.Z);
			this.StartPoint = new CrawlPoint3d(circle.StartPoint.X, circle.StartPoint.Y, circle.StartPoint.Z);
			this.Center = new CrawlPoint3d(circle.Center.X, circle.Center.Y, circle.Center.Z);

			this.Layer = circle.Layer;
			this.Linetype = circle.Linetype;
			this.LineWeight = circle.LineWeight.ToString();
			this.Color = circle.Color.ToString();

			this.Radius = circle.Radius;
			this.Thickness = circle.Thickness;
		}
	}

	[DataContract]
	public class CrawlAcDbEllipse : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbEllipse";

		[DataMember]
		public CrawlPoint3d Center { get; set; }
		[DataMember]
		public CrawlPoint3d StartPoint { get; set; }
		[DataMember]
		public CrawlPoint3d EndPoint { get; set; }

		[DataMember]
		public double Length;

		public CrawlAcDbEllipse() { }

		public CrawlAcDbEllipse(Ellipse ellipse)
		{
			Entity ent = (Entity)ellipse;
			this.ObjectId = ent.ObjectId.ToString();

			this.EndPoint = new CrawlPoint3d(ellipse.EndPoint.X, ellipse.EndPoint.Y, ellipse.EndPoint.Z);
			this.StartPoint = new CrawlPoint3d(ellipse.StartPoint.X, ellipse.StartPoint.Y, ellipse.StartPoint.Z);
			this.Center = new CrawlPoint3d(ellipse.Center.X, ellipse.Center.Y, ellipse.Center.Z);

			this.Layer = ellipse.Layer;
			this.Linetype = ellipse.Linetype;
			this.LineWeight = ellipse.LineWeight.ToString();
			this.Color = ellipse.Color.ToString();

		}
	}

	[DataContract]
	public class CrawlAcDbAlignedDimension : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbAlignedDimension";

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

		public CrawlAcDbAlignedDimension() { }

		public CrawlAcDbAlignedDimension(AlignedDimension dim)
		{
			Entity ent = (Entity)dim;
			this.ObjectId = ent.ObjectId.ToString();

			this.XLine1Point = new CrawlPoint3d(dim.XLine1Point.X, dim.XLine1Point.Y, dim.XLine1Point.Z);
			this.XLine2Point = new CrawlPoint3d(dim.XLine2Point.X, dim.XLine2Point.Y, dim.XLine2Point.Z);
			this.DimLinePoint = new CrawlPoint3d(dim.DimLinePoint.X, dim.DimLinePoint.Y, dim.DimLinePoint.Z);
			this.TextPosition = new CrawlPoint3d(dim.TextPosition.X, dim.TextPosition.Y, dim.TextPosition.Z);

			this.Layer = dim.Layer;
			this.Linetype = dim.Linetype;
			this.LineWeight = dim.LineWeight.ToString();
			this.Color = dim.Color.ToString();

			this.DimensionText = dim.DimensionText;
			this.DimensionStyleName = dim.DimensionStyleName;
		}
	}

	[DataContract]
	public class CrawlAcDbRotatedDimension : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbRotatedDimension";

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

		public CrawlAcDbRotatedDimension() { }

		public CrawlAcDbRotatedDimension(RotatedDimension dim)
		{
			Entity ent = (Entity)dim;
			this.ObjectId = ent.ObjectId.ToString();

			this.XLine1Point = new CrawlPoint3d(dim.XLine1Point.X, dim.XLine1Point.Y, dim.XLine1Point.Z);
			this.XLine2Point = new CrawlPoint3d(dim.XLine2Point.X, dim.XLine2Point.Y, dim.XLine2Point.Z);
			this.DimLinePoint = new CrawlPoint3d(dim.DimLinePoint.X, dim.DimLinePoint.Y, dim.DimLinePoint.Z);
			this.TextPosition = new CrawlPoint3d(dim.TextPosition.X, dim.TextPosition.Y, dim.TextPosition.Z);

			this.Layer = dim.Layer;
			this.Linetype = dim.Linetype;
			this.LineWeight = dim.LineWeight.ToString();
			this.Color = dim.Color.ToString();

			this.DimensionText = dim.DimensionText;
			this.DimensionStyleName = dim.DimensionStyleName;
		}
	}

	[DataContract]
	public class CrawlAcDbPoint3AngularDimension : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbPoint3AngularDimension";

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

		public CrawlAcDbPoint3AngularDimension() { }

		public CrawlAcDbPoint3AngularDimension(Point3AngularDimension dim)
		{
			Entity ent = (Entity)dim;
			this.ObjectId = ent.ObjectId.ToString();

			this.XLine1Point = new CrawlPoint3d(dim.XLine1Point.X, dim.XLine1Point.Y, dim.XLine1Point.Z);
			this.XLine2Point = new CrawlPoint3d(dim.XLine2Point.X, dim.XLine2Point.Y, dim.XLine2Point.Z);
			this.CenterPoint = new CrawlPoint3d(dim.CenterPoint.X, dim.CenterPoint.Y, dim.CenterPoint.Z);
			this.TextPosition = new CrawlPoint3d(dim.TextPosition.X, dim.TextPosition.Y, dim.TextPosition.Z);

			this.Layer = dim.Layer;
			this.Linetype = dim.Linetype;
			this.LineWeight = dim.LineWeight.ToString();
			this.Color = dim.Color.ToString();

			this.DimensionText = dim.DimensionText;
			this.DimensionStyleName = dim.DimensionStyleName;
		}
	}

	[DataContract]
	public class CrawlAcDbLineAngularDimension2 : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbLineAngularDimension2";

		[DataMember]
		public CrawlPoint3d XLine1Start { get; set; }
		[DataMember]
		public CrawlPoint3d XLine1End { get; set; }
		[DataMember]
		public CrawlPoint3d XLine2Start { get; set; }
		[DataMember]
		public CrawlPoint3d XLine2End { get; set; }
		[DataMember]
		public CrawlPoint3d ArcPoint { get; set; }
		[DataMember]
		public CrawlPoint3d TextPosition { get; set; }

		[DataMember]
		public string DimensionText;
		[DataMember]
		public string DimensionStyleName;

		public CrawlAcDbLineAngularDimension2() { }

		public CrawlAcDbLineAngularDimension2(LineAngularDimension2 dim)
		{
			Entity ent = (Entity)dim;
			this.ObjectId = ent.ObjectId.ToString();

			this.XLine1Start = new CrawlPoint3d(dim.XLine1Start.X, dim.XLine1Start.Y, dim.XLine1Start.Z);
			this.XLine1End = new CrawlPoint3d(dim.XLine1End.X, dim.XLine1End.Y, dim.XLine1End.Z);
			this.XLine2Start = new CrawlPoint3d(dim.XLine2Start.X, dim.XLine2Start.Y, dim.XLine2Start.Z);
			this.XLine2End = new CrawlPoint3d(dim.XLine2End.X, dim.XLine2End.Y, dim.XLine2End.Z);
			this.ArcPoint = new CrawlPoint3d(dim.ArcPoint.X, dim.ArcPoint.Y, dim.ArcPoint.Z);
			this.TextPosition = new CrawlPoint3d(dim.TextPosition.X, dim.TextPosition.Y, dim.TextPosition.Z);

			this.Layer = dim.Layer;
			this.Linetype = dim.Linetype;
			this.LineWeight = dim.LineWeight.ToString();
			this.Color = dim.Color.ToString();

			this.DimensionText = dim.DimensionText;
			this.DimensionStyleName = dim.DimensionStyleName;
		}
	}

	[DataContract]
	public class CrawlAcDbDiametricDimension : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbDiametricDimension";

		[DataMember]
		public CrawlPoint3d FarChordPoint { get; set; }
		[DataMember]
		public CrawlPoint3d ChordPoint { get; set; }
		[DataMember]
		public CrawlPoint3d TextPosition { get; set; }

		[DataMember]
		public string DimensionText;
		[DataMember]
		public string DimensionStyleName;

		public CrawlAcDbDiametricDimension() { }

		public CrawlAcDbDiametricDimension(DiametricDimension dim)
		{
			Entity ent = (Entity)dim;
			this.ObjectId = ent.ObjectId.ToString();

			this.FarChordPoint = new CrawlPoint3d(dim.FarChordPoint.X, dim.FarChordPoint.Y, dim.FarChordPoint.Z);
			this.ChordPoint = new CrawlPoint3d(dim.ChordPoint.X, dim.ChordPoint.Y, dim.ChordPoint.Z);
			this.TextPosition = new CrawlPoint3d(dim.TextPosition.X, dim.TextPosition.Y, dim.TextPosition.Z);

			this.Layer = dim.Layer;
			this.Linetype = dim.Linetype;
			this.LineWeight = dim.LineWeight.ToString();
			this.Color = dim.Color.ToString();

			this.DimensionText = dim.DimensionText;
			this.DimensionStyleName = dim.DimensionStyleName;
		}
	}

	[DataContract]
	public class CrawlAcDbArcDimension : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbArcDimension";

		[DataMember]
		public CrawlPoint3d XLine1Point { get; set; }
		[DataMember]
		public CrawlPoint3d XLine2Point { get; set; }
		[DataMember]
		public CrawlPoint3d ArcPoint { get; set; }
		[DataMember]
		public CrawlPoint3d TextPosition { get; set; }

		[DataMember]
		public string DimensionText;
		[DataMember]
		public string DimensionStyleName;

		public CrawlAcDbArcDimension() { }

		public CrawlAcDbArcDimension(ArcDimension dim)
		{
			Entity ent = (Entity)dim;
			this.ObjectId = ent.ObjectId.ToString();

			this.XLine1Point = new CrawlPoint3d(dim.XLine1Point.X, dim.XLine1Point.Y, dim.XLine1Point.Z);
			this.XLine2Point = new CrawlPoint3d(dim.XLine2Point.X, dim.XLine2Point.Y, dim.XLine2Point.Z);
			this.ArcPoint = new CrawlPoint3d(dim.ArcPoint.X, dim.ArcPoint.Y, dim.ArcPoint.Z);
			this.TextPosition = new CrawlPoint3d(dim.TextPosition.X, dim.TextPosition.Y, dim.TextPosition.Z);

			this.Layer = dim.Layer;
			this.Linetype = dim.Linetype;
			this.LineWeight = dim.LineWeight.ToString();
			this.Color = dim.Color.ToString();

			this.DimensionText = dim.DimensionText;
			this.DimensionStyleName = dim.DimensionStyleName;
		}
	}

	[DataContract]
	public class CrawlAcDbRadialDimension : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbRadialDimension";

		[DataMember]
		public CrawlPoint3d Center { get; set; }
		[DataMember]
		public CrawlPoint3d ChordPoint { get; set; }
		[DataMember]
		public CrawlPoint3d TextPosition { get; set; }

		[DataMember]
		public string DimensionText;
		[DataMember]
		public string DimensionStyleName;

		public CrawlAcDbRadialDimension() { }

		public CrawlAcDbRadialDimension(RadialDimension dim)
		{
			Entity ent = (Entity)dim;
			this.ObjectId = ent.ObjectId.ToString();

			this.Center = new CrawlPoint3d(dim.Center.X, dim.Center.Y, dim.Center.Z);
			this.ChordPoint = new CrawlPoint3d(dim.ChordPoint.X, dim.ChordPoint.Y, dim.ChordPoint.Z);
			this.TextPosition = new CrawlPoint3d(dim.TextPosition.X, dim.TextPosition.Y, dim.TextPosition.Z);

			this.Layer = dim.Layer;
			this.Linetype = dim.Linetype;
			this.LineWeight = dim.LineWeight.ToString();
			this.Color = dim.Color.ToString();

			this.DimensionText = dim.DimensionText;
			this.DimensionStyleName = dim.DimensionStyleName;
		}
	}

	[DataContract]
	public class CrawlAcDbHatch : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbHatch";

		[DataMember]
		public double Area;
		[DataMember]
		public string PatternName;

		[DataMember]
		public List<CrawlAcDbPolyline> Loops;

		public CrawlAcDbHatch() { }

		public CrawlAcDbHatch(Hatch hatch)
		{
			Entity ent = (Entity)hatch;
			this.ObjectId = ent.ObjectId.ToString();

			this.Area = hatch.Area;
			this.Layer = hatch.Layer;
			this.Linetype = hatch.Linetype;
			this.LineWeight = hatch.LineWeight.ToString();
			this.Color = hatch.Color.ToString();

			this.PatternName = hatch.PatternName;

			Loops = HatchToPolylines(hatch.ObjectId);
		}

		/// <summary>
		///  Функция преобразования координат контура штриховки. Последовательно пробегает по каждому из контуров штриховки и преобразует их в полилинии
		/// </summary>
		/// <param name="hatchId">ObjectId штриховки Hatch</param>
		/// <returns>Список crawlAcDbPolyline - перечень контуров штриховки</returns>
		private List<CrawlAcDbPolyline> HatchToPolylines(ObjectId hatchId)
		{
			List<CrawlAcDbPolyline> result = new List<CrawlAcDbPolyline>();

			//Исходный код для AutoCAD .Net
			//http://forums.autodesk.com/t5/NET/Restore-hatch-boundaries-if-they-have-been-lost-with-NET/m-p/3779514#M33429

			try
			{

				Hatch hatch = (Hatch)hatchId.GetObject(OpenMode.ForRead);
				if (hatch != null)
				{
					int nLoops = hatch.NumberOfLoops;
					for (int i = 0; i < nLoops; i++)
					{//Цикл по каждому из контуров штриховки
					 //Проверяем что контур является полилинией
						HatchLoop loop = hatch.GetLoopAt(i);
						if (loop.IsPolyline)
						{
							using (Polyline poly = new Polyline())
							{
								//Создаем полилинию из точек контура
								int iVertex = 0;
								foreach (BulgeVertex bv in loop.Polyline)
								{
									poly.AddVertexAt(iVertex++, bv.Vertex, bv.Bulge, 0.0, 0.0);
								}
								result.Add(new CrawlAcDbPolyline(poly));
							}
						}
						else
						{//Если не удалось преобразовать контур к полилинии
						 //Выводим сообщение в командную строку
							CrawlDebug.WriteLine("Ошибка обработки: Контур штриховки - не полилиния");
							//Не будем брать исходный код для штриховок, контур который не сводится к полилинии
						}
					}
				}
			}
			catch (Exception e)
			{
				CrawlDebug.WriteLine("Ошибка обработки штриховки: {0}", e.Message);
			}
			return result;
		}
	}

	[DataContract]
	public class CrawlAcDbSpline : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbSpline";

		[DataMember]
		public double Area;

		[DataMember]
		public List<CrawlPoint3d> Vertixes;

		public CrawlAcDbSpline() { }

		public CrawlAcDbSpline(Spline spline)
		{
			Entity ent = (Entity)spline;
			this.ObjectId = ent.ObjectId.ToString();

			this.Area = spline.Area;

			this.Layer = spline.Layer;
			this.Linetype = spline.Linetype;
			this.LineWeight = spline.LineWeight.ToString();
			this.Color = spline.Color.ToString();

			Vertixes = GetSplinePoints(spline);
		}

		private List<CrawlPoint3d> GetSplinePoints(Spline spline)
		{
			List<CrawlPoint3d> result = new List<CrawlPoint3d>();

			//Исходный пример из AutoCAD:
			//http://through-the-interface.typepad.com/through_the_interface/2007/04/iterating_throu.html
			//сильно в нем не разбирался, просто адаптирован.

			try
			{

				// Количество контрольных точек сплайна
				int vn = spline.NumControlPoints;

				//Цикл по всем контрольным точкам сплайна
				for (int i = 0; i < vn; i++)
				{
					// Could also get the 3D point here
					Point3d pt = spline.GetControlPointAt(i);

					result.Add(new CrawlPoint3d(pt.X, pt.Y, pt.Z));
				}
			}
			catch
			{
				CrawlDebug.WriteLine("Not a spline or something wrong");
			}
			return result;
		}
	}

	[DataContract]
	public class CrawlAcDbPoint : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbPoint";

		[DataMember]
		public CrawlPoint3d Position { get; set; }

		public CrawlAcDbPoint() { }

		public CrawlAcDbPoint(DBPoint pnt)
		{
			Entity ent = (Entity)pnt;
			this.ObjectId = ent.ObjectId.ToString();

			this.Position = new CrawlPoint3d(pnt.Position.X, pnt.Position.Y, pnt.Position.Z);

			this.Layer = pnt.Layer;
			this.Linetype = pnt.Linetype;
			this.LineWeight = pnt.LineWeight.ToString();
			this.Color = pnt.Color.ToString();
		}
	}

	[DataContract]
	public class CrawlAcDbBlockReference : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbBlockReference";

		[DataMember]
		public CrawlPoint3d Position { get; set; }

		[DataMember]
		public string Name;

		[DataMember]
		public List<CrawlAcDbAttributeReference> Attributes;

		public CrawlAcDbBlockReference() { }

		public CrawlAcDbBlockReference(BlockReference blk)
		{
			Entity ent = (Entity)blk;
			this.ObjectId = ent.ObjectId.ToString();

			this.Position = new CrawlPoint3d(blk.Position.X, blk.Position.Y, blk.Position.Z);

			this.Layer = blk.Layer;
			this.Linetype = blk.Linetype;
			this.LineWeight = blk.LineWeight.ToString();
			this.Color = blk.Color.ToString();

			this.Name = blk.Name;

			Attributes = new List<CrawlAcDbAttributeReference>();

			//http://through-the-interface.typepad.com/through_the_interface/2007/07/updating-a-spec.html
			foreach (ObjectId attId in blk.AttributeCollection)
			{
				AttributeReference attRef = (AttributeReference)attId.GetObject(OpenMode.ForRead);
				this.Attributes.Add(new CrawlAcDbAttributeReference(attRef));
			}
		}
	}

	[DataContract]
	public class CrawlAcDbAttributeReference : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbAttributeReference";

		[DataMember]
		public string Tag;
		[DataMember]
		public string TextString;

		public CrawlAcDbAttributeReference() { }

		public CrawlAcDbAttributeReference(AttributeReference attRef)
		{
			Entity ent = (Entity)attRef;
			this.ObjectId = ent.ObjectId.ToString();

			this.Tag = attRef.Tag;
			this.TextString = attRef.TextString;
			this.Color = attRef.Color.ToString();
		}
	}

	[DataContract]
	public class CrawlAcDbProxyEntity : CrawlEntity
	{
		[DataMember]
		public string ClassName = "AcDbProxyEntity";

		[DataMember]
		public string BlockId;
		[DataMember]
		public string FileId;

		public CrawlAcDbProxyEntity(ProxyEntity prxy)
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

		public CrawlAcDbLayerTableRecord() { }

		public CrawlAcDbLayerTableRecord(LayerTableRecord layerRecord)
		{
			Name = layerRecord.Name;

			LinetypeTableRecord ltRec = (LinetypeTableRecord)layerRecord.LinetypeObjectId.GetObject(OpenMode.ForRead);
			this.Linetype = ltRec.Name;

			this.LineWeight = layerRecord.LineWeight.ToString();
			this.IsFrozen = layerRecord.IsFrozen;
			this.IsHidden = layerRecord.IsHidden;
			this.IsOff = layerRecord.IsOff;
			this.IsPlottable = layerRecord.IsPlottable;
			this.Color = layerRecord.Color.ToString();

			this.ObjectId = layerRecord.ObjectId.ToString();
		}

	}
}