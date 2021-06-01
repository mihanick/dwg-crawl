using Multicad;
using Multicad.DatabaseServices;
using Multicad.DatabaseServices.StandardObjects;
using Multicad.Dimensions;
using Multicad.Geometry;
using Multicad.Symbols;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;

namespace DwgDump.Enitites
{
	public static class Converters
	{
		public static List<string> ConvertEntities2json(IEnumerable<McObjectId> entityIds, string fileId, string groupId)
		{
			// Convert nested block structures to flat (we remember blockIds!)

			var jsons = new List<string>();

			var resConvert = Convert(entityIds);
			foreach (var crawlEnt in resConvert)
			{
				crawlEnt.FileId = fileId;
				crawlEnt.GroupId = groupId;

				jsons.Add(Serialize(crawlEnt));

				if (crawlEnt is BlockReference bref)
					foreach (var subEnt in bref.Contents)
					{
						subEnt.FileId = fileId;
						subEnt.GroupId = groupId;
						jsons.Add(Serialize(subEnt));
					}
			}

			return jsons;
		}

		public static IEnumerable<CrawlEntity> Convert(IEnumerable<McObjectId> entityIds)
		{
			var res = new List<CrawlEntity>();
			foreach (var id in entityIds)
			{
				var ent = From(id);
				if (ent != null)
					res.Add(ent);
			}

			return res;
		}

		public static CrawlEntity From(McObjectId id)
		{
			var mcObj = id.GetObject();

			CrawlEntity res = null;

			if (mcObj is DbLine line)
			{
				res = From(line.Line);
			}
			else if (mcObj is DbPolyline pline)
			{
				res = From(pline.Polyline);
			}
			else if (mcObj is DbText dbtxt)
			{
				res = From(dbtxt.Text);
			}
			else if (mcObj is DbCircArc arc)
			{
				res = From(arc.Arc);
			}
			else if (mcObj is DbSpline spline)
			{
				res = From(spline.Spline);
			}
			else if (mcObj is DbCircle circle)
			{
				// Окружность
				CrawlCircle cCircle = new CrawlCircle()
				{
					Center = Pt(circle.Center),
					Radius = circle.Radius
				};

				res = cCircle;
			}
			else if (mcObj is DbEllipticArc ellipse)
			{
				// Эллипс
				Crawlellipse cEll = new Crawlellipse()
				{
					EndPoint = new CrawlPoint3d(ellipse.EllipArc.EndPoint.X, ellipse.EllipArc.EndPoint.Y, ellipse.EllipArc.EndPoint.Z),
					StartPoint = new CrawlPoint3d(ellipse.EllipArc.StartPoint.X, ellipse.EllipArc.StartPoint.Y, ellipse.EllipArc.StartPoint.Z),
					Center = new CrawlPoint3d(ellipse.EllipArc.Center.X, ellipse.EllipArc.Center.Y, ellipse.EllipArc.Center.Z),

					MajorAxisVector = new CrawlPoint3d(ellipse.EllipArc.MajorAxis.X, ellipse.EllipArc.MajorAxis.Y, ellipse.EllipArc.MajorAxis.Z),
					MinorAxisVector = new CrawlPoint3d(ellipse.EllipArc.MinorAxis.X, ellipse.EllipArc.MinorAxis.Y, ellipse.EllipArc.MinorAxis.Z)

				};
				res = cEll;
			}
			else if (mcObj is DbPoint pnt)
			{
				CrawlAcDbPoint cpt = new CrawlAcDbPoint()
				{
					Position = new CrawlPoint3d(pnt.Position.X, pnt.Position.Y, pnt.Position.Z)
				};
				res = cpt;
			}
			else if (mcObj is McLinearDimension dim)
			{
				// Размер повернутый
				var x1 = dim.GetPosition(0);
				var x2 = dim.GetPosition(1);
				LinearDimension cDim = new LinearDimension()
				{
					XLine1Point = new CrawlPoint3d(x1.X, x1.Y, x1.Z),
					XLine2Point = new CrawlPoint3d(x2.X, x2.Y, x2.Z),
					DimLinePoint = new CrawlPoint3d(dim.LinePosition.X, dim.LinePosition.Y, dim.LinePosition.Z),
					TextPosition = new CrawlPoint3d(dim.TextPosition.X, dim.TextPosition.Y, dim.TextPosition.Z),
					DimensionText = dim.Text,
					DimensionStyleName = dim.DimensionStyle.ToString()
				};
				res = cDim;
			}
			else if (mcObj is McAngularDimension dima)
			{
				// Угловой размер по 3 точкам
				AngularDimension cDim = new AngularDimension()
				{
					XLine1Point = new CrawlPoint3d(dima.GetPosition(0).X, dima.GetPosition(0).Y, dima.GetPosition(0).Z),
					XLine2Point = new CrawlPoint3d(dima.GetPosition(1).X, dima.GetPosition(1).Y, dima.GetPosition(1).Z),
					CenterPoint = new CrawlPoint3d(dima.Center.X, dima.Center.Y, dima.Center.Z),
					TextPosition = new CrawlPoint3d(dima.TextPosition.X, dima.TextPosition.Y, dima.TextPosition.Z),

					DimensionText = dima.Text,
					DimensionStyleName = dima.DimensionStyle.ToString()

				};
				res = cDim;
			}
			else if (mcObj is McDiametralDimension dimd)
			{
				// Размер диаметра окружности
				DiameterDimension cDim = new DiameterDimension()
				{
					ArcStartAngle = dimd.ArcStartAngle,
					ArcEndAngle = dimd.ArcEndAngle,
					Center = new CrawlPoint3d(dimd.Center.X, dimd.Center.Y, dimd.Center.Z),
					Pos1 = new CrawlPoint3d(dimd.GetPosition(0).X, dimd.GetPosition(0).Y, dimd.GetPosition(0).Z),

					TextPosition = new CrawlPoint3d(dimd.TextPosition.X, dimd.TextPosition.Y, dimd.TextPosition.Z),
					DimensionText = dimd.Text,

					DimensionStyleName = dimd.DimensionStyle.ToString()
				};
				res = cDim;
			}
			else if (mcObj is McArcDimension dimArc)
			{
				// Дуговой размер
				ArcDimension cDim = new ArcDimension()
				{

					Radius = dimArc.Radius,
					Center = new CrawlPoint3d(dimArc.Center.X, dimArc.Center.Y, dimArc.Center.Z),

					TextPosition = new CrawlPoint3d(dimArc.TextPosition.X, dimArc.TextPosition.Y, dimArc.TextPosition.Z),

					DimensionText = dimArc.Text,
					DimensionStyleName = dimArc.DimensionStyle.ToString()
				};
				res = cDim;
			}
			else if (mcObj is McRadialDimension dimr)
			{
				// Радиальный размер
				RadialDimension cDim = new RadialDimension()
				{
					ArcEndAngle = dimr.ArcEndAngle,
					ArcStartAngle = dimr.ArcEndAngle,
					Radius = dimr.Radius,

					Center = new CrawlPoint3d(dimr.Center.X, dimr.Center.Y, dimr.Center.Z),

					Position = new CrawlPoint3d(dimr.Position.X, dimr.Position.Y, dimr.Position.Z),

					TextPosition = new CrawlPoint3d(dimr.TextPosition.X, dimr.TextPosition.Y, dimr.TextPosition.Z),

					DimensionText = dimr.Text,
					DimensionStyleName = dimr.DimensionStyle.ToString()

				};
				res = cDim;
			}
			else if (mcObj is McConnectionBreak br)
			{
				Break cBreak = new Break()
				{
					StartPoint = new CrawlPoint3d(br.StartPoint.X, br.StartPoint.Y, br.StartPoint.Z),
					EndPoint = new CrawlPoint3d(br.EndPoint.X, br.EndPoint.Y, br.EndPoint.Z)
				};
				res = cBreak;
			}
			else if (mcObj is McNotePosition mcNote)
			{
				var cNote = new CrawlNote()
				{
					FirstLine = mcNote.FirstLine,
					SecondLine = mcNote.SecondLine,
					Origin = new CrawlPoint3d(mcNote.Origin.X, mcNote.Origin.Y, mcNote.Origin.Z),
					ClassName = "McNotePosition"
				};

				for (int i = 0; i < mcNote.Leader.Childs.Count; i++)
				{
					var child = mcNote.Leader.Childs[i];
					CrawlPoint3d start = new CrawlPoint3d(child.Start.X, child.Start.Y, child.Start.Y);
					CrawlPoint3d end = new CrawlPoint3d(child.End.X, child.End.Y, child.End.Y);
					cNote.Lines.Add(new CrawlLine(start, end));
				}

				res = cNote;
			}
			else if (mcObj is McNoteComb combNote)
			{
				var cNote = new CrawlNote()
				{
					FirstLine = combNote.FirstLine,
					SecondLine = combNote.SecondLine,
					Origin = new CrawlPoint3d(combNote.Start.X, combNote.Start.Y, combNote.Start.Z),
					ClassName = "McNoteComb"
				};

				for (int i = 0; i < combNote.Leader.Childs.Count; i++)
				{
					var child = combNote.Leader.Childs[i];
					CrawlPoint3d start = new CrawlPoint3d(child.Start.X, child.Start.Y, child.Start.Y);
					CrawlPoint3d end = new CrawlPoint3d(child.End.X, child.End.Y, child.End.Y);
					cNote.Lines.Add(new CrawlLine(start, end));
				}

				res = cNote;
			}
			else if (mcObj is McNoteChain chainNote)
			{
				var cNote = new CrawlNote()
				{
					FirstLine = chainNote.FirstLine,
					SecondLine = chainNote.SecondLine,
					Origin = new CrawlPoint3d(chainNote.Origin.X, chainNote.Origin.Y, chainNote.Origin.Z),
					ClassName = "McNoteChain"
				};

				for (int i = 0; i < chainNote.Leader.Childs.Count; i++)
				{
					var child = chainNote.Leader.Childs[i];
					CrawlPoint3d start = new CrawlPoint3d(child.Start.X, child.Start.Y, child.Start.Y);
					CrawlPoint3d end = new CrawlPoint3d(child.End.X, child.End.Y, child.End.Y);
					cNote.Lines.Add(new CrawlLine(start, end));
				}
				res = cNote;
			}
			else if (mcObj is McNoteKnot mcNoteKnot)
			{
				var cNote = new CrawlNote()
				{
					FirstLine = mcNoteKnot.Knot,
					SecondLine = mcNoteKnot.Note,
					Origin = new CrawlPoint3d(mcNoteKnot.Center.X, mcNoteKnot.Center.Y, mcNoteKnot.Center.Z),
					ClassName = "McNoteKnot"
				};

				res = cNote;
			}
			else if (mcObj is McNoteLinearMark mcNoteL)
			{
				var cNote = new CrawlNote()
				{
					FirstLine = mcNoteL.FirstLine,
					SecondLine = mcNoteL.SecondLine,
					ClassName = "McNoteLinear"
				};
				cNote.Lines.Add(
					new CrawlLine(
						new CrawlPoint3d(mcNoteL.FirstPnt.X, mcNoteL.FirstPnt.Y, mcNoteL.FirstPnt.Z),
						new CrawlPoint3d(mcNoteL.SecondPnt.X, mcNoteL.SecondPnt.Y, mcNoteL.SecondPnt.Z)
					)
				);

				res = cNote;
			}
			else if (mcObj is McSectionVS section)
			{
				var cSection = new Section()
				{
					Name = section.Word,
					Vertices = section.Points
						.Select(pt => new CrawlPoint3d(pt.X, pt.Y, pt.Z))
						.ToList(),
					ClassName = "Section"
				};

				res = cSection;
			}
			else if (mcObj is McSymbolVS vs)
			{
				var cText = new CrawlText()
				{
					TextString = vs.Word,
					Position = new CrawlPoint3d(vs.Origin.X, vs.Origin.Y, vs.Origin.Z),
					ClassName = "ViewDesignation"
				};
				res = cText;
			}
			else if (mcObj is McConnectionFix fx)
			{
				CrawlPoint3d start = new CrawlPoint3d(fx.BasePoint.X, fx.BasePoint.Y, fx.BasePoint.Z);
				CrawlPoint3d end = new CrawlPoint3d(fx.FixPoint.X, fx.FixPoint.Y, fx.FixPoint.Y);

				var cNote = new CrawlNote()
				{
					FirstLine = fx.StrAbove,
					SecondLine = fx.StrUnder,
					Origin = start,
					ClassName = "WeldDesignation"
				};

				cNote.Lines.Add(new CrawlLine(start, end));

				res = cNote;
			}
			else if (mcObj is McRange rng)
			{
				var cRange = new CrawlPolyline()
				{
					ClassName = "RangeDistributionDesignation"
				};

				cRange.Vertices.Add(new CrawlPoint3d(rng.FirstBoundPosition.X, rng.FirstBoundPosition.Y, rng.FirstBoundPosition.Z));
				cRange.Vertices.Add(new CrawlPoint3d(rng.BasePosition.X, rng.BasePosition.Y, rng.BasePosition.Z));
				cRange.Vertices.Add(new CrawlPoint3d(rng.SecondBoundPosition.X, rng.SecondBoundPosition.Y, rng.SecondBoundPosition.Z));

				res = cRange;
			}
			else if (mcObj is McAxisEntity ax)
			{
				var aaa = (McBasicAxis.McAxisSpecificLinear)ax.Axis;

				var cAxis = new AxisLinear()
				{
					Name = aaa.Markers.First.Value,
					StartPoint = new CrawlPoint3d(aaa.StartPoint.X, aaa.StartPoint.Y, aaa.StartPoint.Z),
					EndPoint = new CrawlPoint3d(aaa.EndPoint.X, aaa.EndPoint.Y, aaa.EndPoint.Z)
				};

				res = cAxis;
			}

			else if (mcObj is McBlockRef blk)
			{
				var block = new BlockReference()
				{
					Name = blk.BlockName,
					Position = Pt(blk.InsertPoint)
				};
				res = block;

				// We will explode blocks and complex entities
				var blockContents = blk.DbEntity.Explode();
				block.Contents = ConvertComplexEntity(blockContents, block.ObjectId);
			}
			else if (mcObj.ClassID.ToString() == "a9b900a6-1b65-4f9c-bee9-d5751a9c4484")
			{
				var dbE = mcObj.Cast<McDbEntity>();
				var mcPs = mcObj.Cast<McPropertySource>();
				var txt = mcPs.ObjectProperties[""].ToString();

				var LevelNoteEnity = new CrawlText()
				{
					ClassName = "LevelMark",
					Position = Pt(dbE.BoundingBox.BasePoint),
					TextString = txt
				};
			}
			else if (mcObj is McEntity enti)
			{
				if (notImplementedGuids.Contains(mcObj.ClassID.ToString()))
					return null;

				var cplxEntity = new BlockReference()
				{
					Name = enti.ClassID.ToString(),
					ClassName = "Entity"
				};

				res = cplxEntity;

				// DEBUG: McObjectManager.SelectionSet.SetSelection(ent.ID);
				// DEBUG: throw new NotImplementedException(ent.ClassID.ToString());
				CrawlDebug.WriteLine("[Warning] Object is exploded: " + mcObj.GetType().Name + mcObj.ClassID.ToString());

				var geom = enti.GeometryCache;
				if (geom.Count == 0)
				{
					// Try to explode entity
					geom = enti.DbEntity.Explode();
				}

				cplxEntity.Contents = ConvertComplexEntity(geom, cplxEntity.ObjectId);
			}
			else
			{
				// Maybe we missed something than we're here
				McObjectManager.SelectionSet.SetSelection(id);
				CrawlDebug.WriteLine("[TODO] Не могу обработать тип объекта: " + mcObj.ToString());

				// DEBUG: throw new NotImplementedException();
			}

			var dbe = mcObj?.Cast<McEntity>()?.DbEntity;

			if (res != null)
				if (dbe != null)
				{
					res.Handle = dbe.ID.Handle;
					res.Color = dbe.Color.ToString();
					res.Layer = dbe.Layer;
					res.Linetype = dbe.LineType.ToString();
					res.LineWeight = dbe.LineWeight.ToString();
					res.ObjectId = id.ToString();
				}
			return res;
		}

		public static CrawlEntity From(EntityGeometry eg)
		{
			CrawlEntity res = null;

			switch (eg.GeometryType)
			{
				case EntityGeomType.kHatch:
					var hatch = eg.HatchGeom;
					var cHatch = new CrawlHatch()
					{
						PatternName = hatch.PatternName,
						Loops = hatch.Contours
							.Select(contour => new CrawlPolyline()
							{
								Vertices =
									contour.Points
										.Select(p => new CrawlPoint3d(p.X, p.Y, p.Z))
										.ToList()
							})
							.ToList()
					};
					res = cHatch;
					break;
				case EntityGeomType.kLine:
					CrawlLine cline = new CrawlLine()
					{
						EndPoint = new CrawlPoint3d(eg.LineSeg.EndPoint.X, eg.LineSeg.EndPoint.Y, eg.LineSeg.EndPoint.Z),
						StartPoint = new CrawlPoint3d(eg.LineSeg.StartPoint.X, eg.LineSeg.StartPoint.Y, eg.LineSeg.StartPoint.Z)
					};

					res = cline;
					break;
				case EntityGeomType.kPolyline:
					CrawlPolyline pline = new CrawlPolyline()
					{
						Vertices =
							eg.Polyline.Points
								.Select(p => new CrawlPoint3d(p.X, p.Y, p.Z))
								.ToList()
					};
					res = pline;
					break;
				case EntityGeomType.kCircArc:
					CrawlArc cArc = new CrawlArc()
					{
						EndPoint = new CrawlPoint3d(eg.CircArc.EndPoint.X, eg.CircArc.EndPoint.Y, eg.CircArc.EndPoint.Z),
						StartPoint = new CrawlPoint3d(eg.CircArc.StartPoint.X, eg.CircArc.StartPoint.Y, eg.CircArc.StartPoint.Z),
						Center = new CrawlPoint3d(eg.CircArc.Center.X, eg.CircArc.Center.Y, eg.CircArc.Center.Z),

						Radius = eg.CircArc.Radius
					};
					res = cArc;
					break;
				case EntityGeomType.kText:
					CrawlText ctext = new CrawlText()
					{
						Position = new CrawlPoint3d(eg.Text.Origin.X, eg.Text.Origin.Y, eg.Text.Origin.Z),
						TextString = eg.Text.Text
					};
					res = ctext;
					break;
				case EntityGeomType.kSpline:
					// Сплайн
					var spline = eg.Nurb;
					Spline cScpline = new Spline();

					for (int i = 0; i < spline.NumFitPoints; i++)
						if (spline.GetFitPointAt(i, out Point3d pnt))
							cScpline.Vertices.Add(new CrawlPoint3d(pnt.X, pnt.Y, pnt.Z));

					for (int i = 0; i < spline.NumControlPoints; i++)
					{
						var pnt = spline.ControlPointAt(i);
						cScpline.ControlPoints.Add(new CrawlPoint3d(pnt.X, pnt.Y, pnt.Z));
					}

					res = cScpline;
					break;
			}

			if (res != null)
			{
				res.Layer = eg.Layer;
				// not required res.ObjectId = 
				res.Linetype = eg.LineType.ToString();
				// not implemented res.LineWeight = eg.LineWeight;
				res.Color = eg.Color.ToString();
			}

			return res;
		}

		private static List<CrawlEntity> ConvertComplexEntity(List<EntityGeometry> eg, string blockId)
		{
			var result = new List<CrawlEntity>();
			foreach (EntityGeometry blockOneEntityGeometry in eg)
			{
				var ent = From(blockOneEntityGeometry);
				if (ent != null)
				{
					ent.BlockId = blockId;
					result.Add(ent);
				}
			}

			return result;
		}

		[DebuggerStepThrough]
		public static string Serialize(object o)
		{
			return JsonConvert.SerializeObject(o);
		}

		/// <summary>
		/// Converts Multicad Point3d to CrawlPoint3d
		/// </summary>
		/// <param name=""></param>
		[DebuggerStepThrough]
		private static CrawlPoint3d Pt(Point3d mpt)
		{
			if (mpt == null)
				return new CrawlPoint3d();

			return new CrawlPoint3d(mpt.X, mpt.Y, mpt.Z);
		}

		private static readonly List<string> notImplementedGuids = new List<string>()
					{
						"a9b900a6-1b65-4f9c-bee9-d5751a9c4484", // Level mark not implemented in API
						"592b8316-9dc5-4c8a-98be-6aaef93747a1", // level mark anchor
						"e02ceca0-2e91-4b7d-9a69-e3e9ebd027c6" // Solid
					};
	}
}