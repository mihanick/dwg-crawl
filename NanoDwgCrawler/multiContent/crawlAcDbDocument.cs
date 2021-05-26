using DwgDump.Data;
using DwgDump.Enitites;
using DwgDump.Util;
using Multicad;
using Multicad.DatabaseServices;
using Multicad.DatabaseServices.StandardObjects;
using Multicad.Dimensions;
using Multicad.Geometry;
using Multicad.Symbols;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace DwgDump
{
	class CrawlAcDbDocument
	{
		private readonly string fullPath;
		private readonly string fileId;

		private readonly McDocument Document;

		private DbMongo Db => DbMongo.Instance;

		// https://stackoverflow.com/questions/2987559/check-if-a-file-is-open
		private bool IsFileinUse(FileInfo file)
		{
			FileStream stream = null;

			try
			{
				stream = file.Open(FileMode.Open, FileAccess.ReadWrite, FileShare.None);
			}
			catch (IOException)
			{
				//the file is unavailable because it is:
				//still being written to
				//or being processed by another thread
				//or does not exist (has already been processed)
				return true;
			}
			finally
			{
				if (stream != null)
					stream.Close();
			}

			return false;
		}

		public CrawlAcDbDocument(CrawlDocument crawlDoc)
		{
			this.fullPath = crawlDoc.Path;
			this.fileId = crawlDoc.FileId;
			string path = Path.Combine(DbMongo.Instance.DataDir, crawlDoc.FileId + ".dwg");
			FileInfo info = new FileInfo(path);
			if (!IsFileinUse(info))
			{
				this.Document = McDocumentsManager.OpenDocument(path, true);

				DumpDocumentDescription();
			}
		}

		private void DumpDocumentDescription()
		{
			try
			{
				// also run all blocks
				var ids = this.Document.GetBlocks();
				// TODO: Maybe we don't need it for now

				// all layer definitions
				foreach (var layername in McObjectManager.CurrentStyle.GetLayers())
				{
					var layerRecord = McObjectManager.CurrentStyle.GetLayer(layername);

					string objId = layerRecord.ToString();

					CrawlAcDbLayerTableRecord cltr = new CrawlAcDbLayerTableRecord()
					{
						Name = layerRecord.Name,

						// BUG: Not implemented
						// this.Linetype = layerRecord.LineType.Name;

						LineWeight = layerRecord.LineWeight.ToString(),
						IsFrozen = layerRecord.IsFrozen,
						// BUG: Not implemented
						// this.IsHidden = layerRecord.IsHidden;
						IsOff = layerRecord.IsOff,
						IsPlottable = layerRecord.IsPlottable,
						Color = layerRecord.Color.ToString(),

						ObjectId = layerRecord.ID.ToString()
					};

					string objectJson = CrawlJsonHelper.Serialize(cltr);

					this.Db.SaveObjectData(objectJson, this.fileId);
				}

				// Run all xrefs
				List<McObjectId> xrefs = this.Document.GetXRefs();
				// TODO: Beacuse in current dataset no xref will be present
			}
			catch (Exception e)
			{
				CrawlDebug.WriteLine(e.Message);
				// Cannot dump layers and xrefs
			}
		}

		public void DumpEntities()
		{
			// If document wasn't loaded correctly
			if (this.Document == null)
				return;
			// nanoCAD can crash, or exception or whatever...
			// so there will be only one try per document
			// so we first set document scanned, 
			// than try to process it
			Db.SetDocumentScanned(this.fileId);

			var filter = new ObjectFilter();
			filter.AddDoc(this.Document);
			filter.AllObjects = true;

			// каждый набор объектов со своим Guid группы
			var groupId = Guid.NewGuid().ToString();
			DumpEntities(filter.GetObjects(), groupId);
		}

		public void DumpEntities(List<McObjectId> entityIds, string groupId)
		{
			this.Db.SaveObjectData(ConvertEntities2json(entityIds), this.fileId, groupId);
		}

		private string ConvertSingleEntity2json(McObject mcObj)
		{
			string result = "";

			// Check primitive is a single entity

			if (mcObj is DbLine line)
			{
				result = ConvertSingleEntityGeometry(line.Line);
			}
			else if (mcObj is DbPolyline pline)
			{
				result = ConvertSingleEntityGeometry(pline.Polyline);
			}
			else if (mcObj is DbText dbtxt)
			{
				result = ConvertSingleEntityGeometry(dbtxt.Text);
			}
			else if (mcObj is DbCircArc arc)
			{
				result = ConvertSingleEntityGeometry(arc.Arc);
			}
			else if (mcObj is DbCircle circle)
			{
				// Окружность
				CrawlCircle cCircle = new CrawlCircle()
				{
					Center = new CrawlPoint3d(circle.Center.X, circle.Center.Y, circle.Center.Z),
					Radius = circle.Radius
				};
				result = CrawlJsonHelper.Serialize(cCircle);
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
				result = CrawlJsonHelper.Serialize(cEll);
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
				result = CrawlJsonHelper.Serialize(cDim);
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
				result = CrawlJsonHelper.Serialize(cDim);
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
				result = CrawlJsonHelper.Serialize(cDim);
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
				result = CrawlJsonHelper.Serialize(cDim);
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
				result = CrawlJsonHelper.Serialize(cDim);
			}
			else if (mcObj is DbSpline spline)
			{
				return ConvertSingleEntityGeometry(spline.Spline);
			}
			else if (mcObj is DbPoint pnt)
			{
				CrawlAcDbPoint cpt = new CrawlAcDbPoint()
				{
					Position = new CrawlPoint3d(pnt.Position.X, pnt.Position.Y, pnt.Position.Z)
				};
				result = CrawlJsonHelper.Serialize(cpt);
			}
			else if (mcObj is McConnectionBreak br)
			{
				Break cBreak = new Break()
				{
					StartPoint = new CrawlPoint3d(br.StartPoint.X, br.StartPoint.Y, br.StartPoint.Z),
					EndPoint = new CrawlPoint3d(br.EndPoint.X, br.EndPoint.Y, br.EndPoint.Z)
				};
				result = CrawlJsonHelper.Serialize(cBreak, "Break");
			}
			else if (mcObj is McNotePosition mcNote)
			{
				var cNote = new CrawlNote()
				{
					FirstLine = mcNote.FirstLine,
					SecondLine = mcNote.SecondLine,
					Origin = new CrawlPoint3d(mcNote.Origin.X, mcNote.Origin.Y, mcNote.Origin.Z)
				};

				for (int i = 0; i < mcNote.Leader.Childs.Count; i++)
				{
					var child = mcNote.Leader.Childs[i];
					CrawlPoint3d start = new CrawlPoint3d(child.Start.X, child.Start.Y, child.Start.Y);
					CrawlPoint3d end = new CrawlPoint3d(child.End.X, child.End.Y, child.End.Y);
					cNote.Lines.Add(new CrawlLine(start, end));
				}

				result = CrawlJsonHelper.Serialize(cNote, "McNotePosition");
			}
			else if (mcObj is McNoteComb combNote)
			{
				var cNote = new CrawlNote()
				{
					FirstLine = combNote.FirstLine,
					SecondLine = combNote.SecondLine,
					Origin = new CrawlPoint3d(combNote.Start.X, combNote.Start.Y, combNote.Start.Z)
				};

				for (int i = 0; i < combNote.Leader.Childs.Count; i++)
				{
					var child = combNote.Leader.Childs[i];
					CrawlPoint3d start = new CrawlPoint3d(child.Start.X, child.Start.Y, child.Start.Y);
					CrawlPoint3d end = new CrawlPoint3d(child.End.X, child.End.Y, child.End.Y);
					cNote.Lines.Add(new CrawlLine(start, end));
				}

				result = CrawlJsonHelper.Serialize(cNote, "McNoteComb");
			}
			else if (mcObj is McNoteChain chainNote)
			{
				var cNote = new CrawlNote()
				{
					FirstLine = chainNote.FirstLine,
					SecondLine = chainNote.SecondLine,
					Origin = new CrawlPoint3d(chainNote.Origin.X, chainNote.Origin.Y, chainNote.Origin.Z)
				};

				for (int i = 0; i < chainNote.Leader.Childs.Count; i++)
				{
					var child = chainNote.Leader.Childs[i];
					CrawlPoint3d start = new CrawlPoint3d(child.Start.X, child.Start.Y, child.Start.Y);
					CrawlPoint3d end = new CrawlPoint3d(child.End.X, child.End.Y, child.End.Y);
					cNote.Lines.Add(new CrawlLine(start, end));
				}
				result = CrawlJsonHelper.Serialize(cNote, "McNoteChain");
			}
			else if (mcObj is McNoteKnot mcNoteKnot)
			{
				var cNote = new CrawlNote()
				{
					FirstLine = mcNoteKnot.Knot,
					SecondLine = mcNoteKnot.Note,
					Origin = new CrawlPoint3d(mcNoteKnot.Center.X, mcNoteKnot.Center.Y, mcNoteKnot.Center.Z)
				};

				result = CrawlJsonHelper.Serialize(cNote, "McNoteKnot");
			}
			else if (mcObj is McNoteLinearMark mcNoteL)
			{
				var cNote = new CrawlNote()
				{
					FirstLine = mcNoteL.FirstLine,
					SecondLine = mcNoteL.SecondLine,
				};
				cNote.Lines.Add(
					new CrawlLine(
						new CrawlPoint3d(mcNoteL.FirstPnt.X, mcNoteL.FirstPnt.Y, mcNoteL.FirstPnt.Z),
						new CrawlPoint3d(mcNoteL.SecondPnt.X, mcNoteL.SecondPnt.Y, mcNoteL.SecondPnt.Z)
					)
				);

				result = CrawlJsonHelper.Serialize(cNote, "McNoteLinear");
			}
			else if (mcObj is McSectionVS section)
			{
				var cSection = new Section()
				{
					Name = section.Word,
					Vertices = section.Points
					.Select(pt => new CrawlPoint3d(pt.X, pt.Y, pt.Z))
					.ToList()
				};

				result = CrawlJsonHelper.Serialize(cSection, "Section");
			}
			else if (mcObj is McSymbolVS vs)
			{
				var cText = new CrawlText()
				{
					TextString = vs.Word,
					Position = new CrawlPoint3d(vs.Origin.X, vs.Origin.Y, vs.Origin.Z)
				};
				result = CrawlJsonHelper.Serialize(cText, "ViewDesignation");
			}

			else if (mcObj is McConnectionFix fx)
			{
				CrawlPoint3d start = new CrawlPoint3d(fx.BasePoint.X, fx.BasePoint.Y, fx.BasePoint.Z);
				CrawlPoint3d end = new CrawlPoint3d(fx.FixPoint.X, fx.FixPoint.Y, fx.FixPoint.Y);

				var cNote = new CrawlNote()
				{
					FirstLine = fx.StrAbove,
					SecondLine = fx.StrUnder,
					Origin = start
				};

				cNote.Lines.Add(new CrawlLine(start, end));

				result = CrawlJsonHelper.Serialize(cNote, "WeldDesignation");
			}
			else if (mcObj is McRange rng)
			{
				var cRange = new CrawlPolyline()
				{

				};

				cRange.Vertices.Add(new CrawlPoint3d(rng.FirstBoundPosition.X, rng.FirstBoundPosition.Y, rng.FirstBoundPosition.Z));
				cRange.Vertices.Add(new CrawlPoint3d(rng.BasePosition.X, rng.BasePosition.Y, rng.BasePosition.Z));
				cRange.Vertices.Add(new CrawlPoint3d(rng.SecondBoundPosition.X, rng.SecondBoundPosition.Y, rng.SecondBoundPosition.Z));

				result = CrawlJsonHelper.Serialize(cRange, "RangeDistributionDesignation");
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

				result = CrawlJsonHelper.Serialize(cAxis);
			}

			// Not implemented multicad geometry, like hatch
			else if (mcObj is McEntity enti)
			{
				if (enti.GeometryCache.Count == 1)
				{
					result = ConvertSingleEntityGeometry(enti.GeometryCache[0]);
				}
			}

			// Populate entity with common properties
			if (!string.IsNullOrEmpty(result))
			{
				var mcent = mcObj.Cast<McDbEntity>();
				JObject jo = JObject.Parse(result);
				jo["Layer"] = mcent.Layer;
				jo["ObjectId"] = mcent.ID.ToString();
				jo["Linetype"] = mcent.LineTypeName;
				jo["LineWeight"] = mcent.LineWeight.ToString();
				jo["Color"] = mcent.Color.ToString();
			}

			return result;
		}

		private string ConvertSingleEntityGeometry(EntityGeometry eg)
		{
			var result = string.Empty;

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
					return CrawlJsonHelper.Serialize(cHatch);
				case EntityGeomType.kLine:
					CrawlLine cline = new CrawlLine()
					{
						EndPoint = new CrawlPoint3d(eg.LineSeg.EndPoint.X, eg.LineSeg.EndPoint.Y, eg.LineSeg.EndPoint.Z),
						StartPoint = new CrawlPoint3d(eg.LineSeg.StartPoint.X, eg.LineSeg.StartPoint.Y, eg.LineSeg.StartPoint.Z)
					};

					return CrawlJsonHelper.Serialize(cline);
				case EntityGeomType.kPolyline:
					CrawlPolyline pline = new CrawlPolyline()
					{
						Vertices =
							eg.Polyline.Points
								.Select(p => new CrawlPoint3d(p.X, p.Y, p.Z))
								.ToList()
					};
					return CrawlJsonHelper.Serialize(pline);
				case EntityGeomType.kCircArc:
					CrawlArc cArc = new CrawlArc()
					{
						EndPoint = new CrawlPoint3d(eg.CircArc.EndPoint.X, eg.CircArc.EndPoint.Y, eg.CircArc.EndPoint.Z),
						StartPoint = new CrawlPoint3d(eg.CircArc.StartPoint.X, eg.CircArc.StartPoint.Y, eg.CircArc.StartPoint.Z),
						Center = new CrawlPoint3d(eg.CircArc.Center.X, eg.CircArc.Center.Y, eg.CircArc.Center.Z),

						Radius = eg.CircArc.Radius
					};
					return CrawlJsonHelper.Serialize(cArc);
				case EntityGeomType.kText:
					CrawlText ctext = new CrawlText()
					{
						Position = new CrawlPoint3d(eg.Text.Origin.X, eg.Text.Origin.Y, eg.Text.Origin.Z),
						TextString = eg.Text.Text
					};
					return CrawlJsonHelper.Serialize(ctext);
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

					return CrawlJsonHelper.Serialize(cScpline);
				default:
					return result;
			}
		}

		private List<string> ConvertEntities2json(IEnumerable<McObjectId> entityIds)
		{
			List<string> jsons = new List<string>();
			foreach (var id_platf in entityIds)
			{
				if (id_platf.IsNull)
					return null;

				var ent = id_platf.GetObject();

				// Всякое может случиться
				var singleRes = ConvertSingleEntity2json(ent);

				if (!string.IsNullOrEmpty(singleRes))
					jsons.Add(singleRes);

				else if (ent is McBlockRef blk)
				{
					// Блоки разбиваем
					var blockContents = blk.DbEntity.Explode();
					foreach (EntityGeometry blockOneEntityGeometry in blockContents)
					{
						var blockEntityJson = ConvertSingleEntityGeometry(blockOneEntityGeometry);
						if (!string.IsNullOrEmpty(blockEntityJson))
						{
							var mcent = blk.DbEntity;
							JObject jo = JObject.Parse(blockEntityJson);
							jo["Layer"] = mcent.Layer;
							jo["ObjectId"] = mcent.ID.ToString();
							jo["Linetype"] = mcent.LineTypeName;
							jo["LineWeight"] = mcent.LineWeight.ToString();
							jo["Color"] = mcent.Color.ToString();

							jsons.Add(jo.ToString());
						}
					}
				}
				else if (ent is McEntity enti)
				{
					List<string> notImplementedGuids = new List<string>()
					{
						"a9b900a6-1b65-4f9c-bee9-d5751a9c4484", // Level mark not implemented in API
						"592b8316-9dc5-4c8a-98be-6aaef93747a1", // level mark anchor
						"e02ceca0-2e91-4b7d-9a69-e3e9ebd027c6" // Solid
					};
					if (notImplementedGuids.Contains(ent.ClassID.ToString()))
						continue;

					McObjectManager.SelectionSet.SetSelection(ent.ID);
					CrawlDebug.WriteLine("[TODO] Не могу обработать тип объекта: " + ent.GetType().Name + ent.ClassID.ToString());
					// throw new NotImplementedException(ent.ClassID.ToString());

					var geom = enti.GeometryCache;

					if (geom.Count == 0)
					{
						CrawlDebug.WriteLine("[TODO] Не могу обработать класс объекта: " + ent.ClassID);
						// Try to explode entity
						geom = enti.DbEntity.Explode();
					}

					foreach (var eg in geom)
					{
						var singleGeom = ConvertSingleEntityGeometry(eg);
						if (!string.IsNullOrEmpty(singleGeom))
							jsons.Add(singleGeom);
					}
				}
				else
				{
					// Если объект не входит в число перечисленных типов,
					// то выводим ошибку, выставляем селекцию по объекту

					McObjectManager.SelectionSet.SetSelection(id_platf);
					CrawlDebug.WriteLine("[TODO] Не могу обработать тип объекта: " + ent.ToString());

					throw new NotImplementedException();
				}

				ent.Cast<McDbEntity>().Color = Color.DarkSeaGreen;
			}
			return jsons;
		}
	}
}
